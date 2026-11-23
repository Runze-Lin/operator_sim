"""Verify/export grouped MX receipts without repacking raw CA traces or rerunning CA."""
import argparse
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import subprocess
import time

from calibration.__main__ import digest, encoded, require, sha_field, stage_window
from calibration.msopprof import sha
from calibration.routed_gmm.campaign import CHIPS, DEV, HOLD
from operator_model import estimate

CONTRACT='routed_grouped_matmul_mxfp4_v5'
API='aclnnGroupedMatmulV5'
DEFAULT=Path(__file__).parent/'reference.json.gz'


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def query(shape,attrs,chip):
    start=time.perf_counter_ns()
    result=estimate('routed_grouped_matmul',shape,attrs,dtype='mxfp4',chip=chip)
    return result,(time.perf_counter_ns()-start)/1e6


def pack(root, include_pending=False):
    frozen=read(root/'freeze.json')
    stamp=sha(root/'freeze.json')
    observations=[]
    failures=[]
    for role,cases in [('dev',DEV),('hold',HOLD)]:
        for idx,(counts,n,k) in enumerate(cases):
            for sku,chip in CHIPS.items():
                identity=f'{role}-{sku}-{idx:02d}'
                raw=root/identity
                if not (raw/'receipt.json').is_file():
                    cost=read(raw/'cost.json') if (raw/'cost.json').is_file() else {}
                    if not include_pending:
                        require(cost.get('exit_code') not in (None,0) or cost.get('stop_reason'),
                                f'Unfinished case {identity}')
                    failures.append({'id':identity,'shape':[sum(counts),n,k],'counts':counts,'sku':chip,
                                     'cost':cost,'status':'failed' if cost else 'pending'})
                    continue
                r=read(raw/'receipt.json')
                attrs={'expert_counts':counts,'implementation':API}
                shape=[sum(counts),n,k]
                prediction,ms=query(shape,attrs,chip)
                future,ms970=query(shape,attrs,'Ascend970DT')
                cp={**r}
                cost=cp.pop('cost')
                if cost is None:
                    # Some completed captures have the full parsed log but no
                    # external stopwatch. Report file open-to-last-write to seconds resolution.
                    birth,modified=map(int,subprocess.check_output(
                        ['stat','-c','%W %Y',str(raw/'run.log')],text=True).split())
                    require(modified>birth>0,'Missing development log timestamps')
                    cost={'started_utc':datetime.fromtimestamp(birth,timezone.utc).isoformat(),
                          'wall_seconds':modified-birth,'method':'log creation to final write, one-second resolution'}
                case={'id':'routed-gmm-'+identity,'op':'routed_grouped_matmul',
                      'contract':CONTRACT,'shape':shape,'attrs':attrs,'prefix':None,
                      'dtype':'mxfp4','api':API,'chip':chip,'outputs':{'Y':sum(counts)*n}}
                if role=='hold':
                    frozen_price=frozen['cases'][f'{idx:02d}-{sku}']['prediction']
                    require(cp['freeze_sha256']==stamp and cp['predicted_us']==frozen_price==prediction.latency_us,
                            'Holdout price differs from frozen forecast')
                case.update(predicted_us=prediction.latency_us,extrapolated_970dt_us=future.latency_us,
                            query_ms=ms)
                begin=datetime.fromisoformat(cost['started_utc'])
                when=(begin+timedelta(seconds=cost['wall_seconds'])).isoformat()
                trace=digest(encoded(cp['core_shard_hashes']))
                receipt={'schema':1,'kind':'ca-receipt','complete':True,'case':case,
                         'provider':cp['provider'],'timing':'all-core-instruction',
                         'started_utc':begin.isoformat(),'finished_utc':when,
                         'correctness':{'Y':{'checked':cp['output_checked'],'bad':cp['mismatches']}},
                         'run_log_sha256':cp['log_sha256'],
                         'stages':[{'kernel':cp['kernel'],'kernel_binary_sha256':cp['elf_sha256'],
                                    'trace_sha256':trace,'ticks_per_us':cp['ticks_per_us'],
                                    'launched_cores':sorted(cp['cores']),'cores':cp['cores'],
                                    'physical_shard_sha256':cp['core_shard_hashes']}],
                         'installed_object':cp['binary'],'phases':cp['phases']}
                if role=='hold':receipt['freeze_sha256']=stamp
                measurement={'case_id':case['id'],'data_role':'independent' if role=='hold' else 'replay',
                             'receipt_sha256':digest(encoded(receipt)),'ca_us':cp['ca_us'],
                             'kernel':[cp['kernel']],'provider':cp['provider'],'identity':[trace]}
                observations.append({'receipt':receipt,'measurement':measurement,'wall_seconds':cost['wall_seconds'],
                                     'cost_method':cost.get('method','monotonic stopwatch'),
                                     'probe_sha256':frozen['probe'],'query_970_ms':ms970})
    return {'schema':1,'kind':'routed-gmm-native','freeze':frozen,'freeze_file_sha256':stamp,
            'freeze_source':(root/'freeze.json').read_text(),
            'capture_policy':{'parallel_ca':1,'msopprof_timeout_minutes':15,'wall_timeout_seconds':960,
                              'shared_disk_reserve_gib':75,'first_manual_dev_timeout_minutes':12},
            'observations':observations,'unqualified':failures}


def rows(document):
    frozen=document['freeze']; freeze_stamp=document['freeze_file_sha256']
    require(freeze_stamp==digest(document['freeze_source'].encode()) and frozen==json.loads(document['freeze_source']),
            'Broken frozen file')
    require(all(sha_field(h) for h in frozen['model'].values()) and sha_field(frozen['probe']),'Missing model or driver hash')
    unique=set();targets=set()
    for obs in document['observations']:
        r,m=obs['receipt'],obs['measurement'];c=r['case']; stage,=r['stages']; ph=r['phases']
        rid=c['id']; role,sku,index=rid.removeprefix('routed-gmm-').split('-')
        require(rid==m['case_id'] and rid not in unique and CHIPS[sku]==c['chip'],'Invalid receipt identity')
        unique.add(rid)
        require(c['contract']==CONTRACT and c['api']==API and c['dtype']=='mxfp4' and
                c['attrs']['implementation']==API and c['outputs']=={'Y':c['shape'][0]*c['shape'][1]},
                'Invalid public contract')
        require(obs['probe_sha256']==frozen['probe'] and m['receipt_sha256']==digest(encoded(r)),
                'Broken receipt or driver hash')
        require(r['complete'] and r['correctness']=={'Y':{'checked':c['outputs']['Y'],'bad':0}},
                'Incomplete numerical comparison')
        hashes=stage['physical_shard_sha256']
        require(set(hashes)==set(stage['cores']) and all(sha_field(h) for h in hashes.values())
                and digest(encoded(hashes))==stage['trace_sha256'] and m['identity']==[stage['trace_sha256']]
                and stage['trace_sha256'] not in targets, 'Broken or duplicate physical-core evidence')
        targets.add(stage['trace_sha256'])
        t=stage_window(stage)
        fragments=ph['fragments']
        calculated_mac=sum(f['m']*f['n']*f['k']*f['count'] for f in fragments)
        require(calculated_mac==math.prod(c['shape']) and sum(f['count'] for f in fragments)==ph['phases']['MMAD']['count'],
                'MMAD fragments do not cover the expected local output dots')
        require(math.isclose(t,m['ca_us'],abs_tol=.000001) and abs(t-ph['full_window_us'])<.001,
                'Inconsistent official/physical timing')
        require(m['kernel']==[stage['kernel']] and stage['kernel'].startswith(r['installed_object'].removesuffix('.o')+'_')
                and all(f['a']==f['b']=='fp4_e2m1' for f in ph['fragments']) and ph['phases']['MMAD']['count']>0,
                'Unqualified MX precision or OPP route')
        expected=(DEV if role=='dev' else HOLD)[int(index)]
        require(c['shape']==[sum(expected[0]),expected[1],expected[2]] and c['attrs']['expert_counts']==expected[0],
                'Histogram or matrix shape mismatch')
        start=datetime.fromisoformat(r['started_utc'])
        if role=='hold':
            f=frozen['cases'][f'{index}-{sku}']
            require(m['data_role']=='independent' and start>datetime.fromisoformat(frozen['utc'])
                    and f['counts']==c['attrs']['expert_counts'] and f['shape']==c['shape'] and f['sku']==c['chip']
                    and f['prediction']==c['predicted_us'] and r['freeze_sha256']==freeze_stamp,'Independence failure')
        else:
            require(m['data_role']=='replay' and start<datetime.fromisoformat(frozen['utc']),'Development role mismatch')
        win={k:v for k,v in c.items() if k not in ('id','outputs')}
        yield {**win,**m,'signed_error_pct':100*(c['predicted_us']/t-1),
               'query_970_ms':obs['query_970_ms'],'acquisition_seconds':obs['wall_seconds']}

    expected_names={f'routed-gmm-{role}-{s}-{i:02d}' for s in CHIPS
                    for role,cases in [('dev',DEV),('hold',HOLD)] for i in range(len(cases))}
    pending={'routed-gmm-'+case['id'] for case in document['unqualified']}
    require(unique.isdisjoint(pending) and unique | pending == expected_names
            and len(pending)==len(document['unqualified']), 'Missing, duplicated or unlabelled acquisition')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,help='Create a new archive from local raw receipts')
    p.add_argument('--archive',type=Path,default=DEFAULT)
    p.add_argument('--partial',action='store_true',help='Include pending status for an in-progress audit')
    p.add_argument('--windows',action='store_true')
    args=p.parse_args()
    d=pack(args.raw,args.partial) if args.raw else read(args.archive)
    windows=list(rows(d))
    if args.raw:
        args.archive.write_bytes(gzip.compress(encoded(d)+b'\n',mtime=0))
    if args.windows:
        for r in windows:print(json.dumps(r,ensure_ascii=False))
    else:
        for role in ('replay','independent'):
            for chip in CHIPS.values():
                sample=[r for r in windows if r['data_role']==role and r['chip']==chip]
                if sample:
                    wape=100*sum(abs(r['predicted_us']-r['ca_us']) for r in sample)/sum(r['ca_us'] for r in sample)
                    print(role,chip,len(sample),f"WAPE {wape:.2f}% max {max(abs(r['signed_error_pct']) for r in sample):.2f}%")
        print('unqualified/pending',len(d['unqualified']))


if __name__=='__main__':
    main()
