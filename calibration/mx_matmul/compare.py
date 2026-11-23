"""Compare two frozen MX models on the same qualified, paired projection cases."""
import argparse
import gzip
import json
import math
import re
from pathlib import Path
from statistics import mean
from calibration.__main__ import CASE_FIELDS, digest, encoded, ingest, integer, require, sha_field, stamp, unseal
from .collect import OBJECT
from .projection_cases import SKUS

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1]/'reference/mx-projections.json.gz'
VERSIONS = ('v1.2', 'v1.3')


def load_comparison(path=DEFAULT_ARCHIVE):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


ROUTES = {'matmul_mx_native': ('a4w4','mxfp4',{},'fp4_e2m1'),
          'matmul_mx_mixed': ('a8w4','fp8_e4m3fn',{'weight_dtype':'mxfp4'},'fp8_e4m3')}
# Both SKUs used these CANN 9.2.1 OPP objects in the retained development receipts.
# Pin the comparison to the same build even when the runtime object stem matches.
KERNEL_SHA256 = {'a4w4':'4738d4c61e0f181c1ba45319dafab54adc5c8121e438525d414a5bca2ab38613',
                 'a8w4':'81b4285e09283a53c36a911903065b9a7e1068dfddc6ec92bd9e4ccf352d8249'}


def validate_evidence(receipt, phases, measurement, *, physical=True):
    """Bind the timed kernel and active engines to the frozen precision contract."""
    case = receipt['case']
    require(case['contract'] in ROUTES, 'Unexpected dense MX contract')
    tag,dtype,attrs,mmad_dtype = ROUTES[case['contract']]
    m,n,_ = case['shape']
    require(case['op']=='matmul' and case['api']=='aclnnQuantMatmulV5' and case['prefix'] is None
            and case['dtype']==dtype and case['attrs']==attrs and case['outputs']=={'Y':m*n}
            and case['chip'] in SKUS, 'Precision, output or dispatch contract mismatch')
    require(len(receipt['stages'])==1 and receipt['provider']=='CANN 9.2.1 ops-nn', 'Expected one 9.2.1 library kernel')
    stage = receipt['stages'][0]
    expected_kernel = re.escape(OBJECT[tag][1]) + r'_\d+(?:_mix_ai[cv])?'
    require(re.fullmatch(expected_kernel,stage['kernel']) and stage['kernel_binary_sha256']==KERNEL_SHA256[tag],
            'Runtime kernel disagrees with the pinned library build or precision route')
    roster = stage['cores']
    if physical or 'physical_shard_sha256' in stage:
        shards = stage.get('physical_shard_sha256',{})
        require(shards.keys()==roster.keys() and all(sha_field(v) for v in shards.values())
                and digest(encoded(shards))==stage['trace_sha256'], 'Physical traces differ from timing identity')
    aic = phases['active_mmad_aics']
    aiv = phases['active_vector_instruction_aivs']
    require(phases['clock_zero']=='earliest physical-core event' and phases['physical_cores']==len(roster)
            and abs(phases['full_window_us']-measurement['ca_us'])<.002, 'Phase clock or physical roster mismatch')
    require(integer(aic,1) and aic<=sum('.cubecore' in c for c in roster)
            and integer(aiv) and aiv<=sum('.veccore' in c for c in roster)
            and ((aiv>0) == (tag=='a8w4')), 'Inconsistent active core counts')
    fragments = phases['fragments']
    mmads = phases['phases']['MMAD']
    require(fragments and all(f['a']==mmad_dtype and f['b']==mmad_dtype and integer(f['count'],1)
            and all(integer(f[axis],1) for axis in ('m','n','k')) for f in fragments)
            and sum(f['count'] for f in fragments)==mmads['count']
            and mmads['instruction_names']=={'MMAD_MX':mmads['count']} and mmads['count']>=aic,
            'MMAD fragments, precision or counts differ')
    if tag=='a8w4':
        require(phases['phases'].get('MTE3',{}).get('instruction_names',{}).get('MOV_UB_TO_L1',0)>=aiv,
                'Producer AIV count lacks weight handoff descriptors')


def validate_schedule(plan, chip):
    require(plan['status'] in ('estimated','extrapolated') and math.isfinite(plan['us']) and plan['us']>0
            and integer(plan['aic'],1) and plan['aic']<=chip['cube_cores']
            and integer(plan['launched_aiv']) and plan['launched_aiv']<=chip['vector_cores']
            and integer(plan['producer_aiv']) and plan['producer_aiv']<=plan['launched_aiv'],
            'Unpriced or impossible frozen schedule')
    stages = plan['stages_us']
    require(stages and all(math.isfinite(v) and v>=0 for v in stages.values())
            and math.isclose(sum(stages.values()),plan['us'],rel_tol=1e-12,abs_tol=1e-9),
            'Frozen schedule budget does not reproduce its device window')


def compare(archive):
    plans={}
    frozen={}
    for version in VERSIONS:
        frozen[version]=unseal(archive[version]['freeze'], 'freeze')
        require(re.fullmatch(re.escape(version[1:])+r'\.\d+', frozen[version]['model_version'])
                and sha_field(frozen[version]['package_sha256']), 'Pricing version or source fingerprint mismatch')
        schedule=unseal(archive[version]['schedules'], 'mx-plan-freeze')
        require(schedule['freeze_sha256']==frozen[version]['sha256'], 'Timing and core freezes differ')
        plans[version]=schedule['plans']
    old, new = ([{key:c[key] for key in CASE_FIELDS} for c in frozen[v]['cases']] for v in VERSIONS)
    require(frozen[VERSIONS[0]]['package_sha256']!=frozen[VERSIONS[-1]]['package_sha256'],
            'Different pricing versions cannot use the same frozen source')
    require(old==new and frozen[VERSIONS[0]]['chips']==frozen[VERSIONS[-1]]['chips'],
            'Frozen inputs or chip resource definitions differ')
    cases={c['id']:c for c in new}
    by_id={v:{c['id']:c for c in frozen[v]['cases']} for v in VERSIONS}
    require(len(cases)==len(new) and cases and all(set(plans[v])==set(cases) for v in VERSIONS),
            'Duplicate or missing frozen case or schedule')
    geometries=archive['geometries']
    declared={tuple(g['shape']):g['id'] for g in geometries}
    require(len(declared)==len(geometries)==len(set(declared.values())), 'Duplicate geometry or geometry id')
    budget=archive['budget']
    require(budget['points']==len(cases) and budget['geometries_per_route']==len(geometries)
            and budget['observed_before_new_acquisition']==0, 'Frozen cases differ from the prospective budget')
    expected = {(contract,shape,sku) for shape in declared for contract in ROUTES for sku in SKUS}
    require(len(expected)==len(cases) and expected=={(c['contract'],tuple(c['shape']),c['chip']) for c in new},
            'Both routes and both SKUs are required for every declared geometry')
    groups={}
    seen=set()
    observed={}
    physical_ids=set()
    for rec in archive['observations']:
        receipt=rec['receipt']
        measurement=ingest(frozen[VERSIONS[-1]],receipt)
        ident=measurement['case_id']
        require(ident not in seen and measurement['data_role']=='prospective'
                and all(stamp(frozen[v]['created_utc'])<=stamp(archive[v]['schedules']['created_utc'])
                        <stamp(receipt['started_utc']) for v in VERSIONS),
                'Duplicate or nonprospective observation')
        seen.add(ident)
        phases=rec['phases']
        validate_evidence(receipt,phases,measurement)
        target_chip=frozen[VERSIONS[-1]]['chips'][cases[ident]['chip']]
        require(receipt['stages'][0]['ticks_per_us']==1000*target_chip['frequency_ghz'],
                'Receipt clock differs from the frozen chip')
        require(not physical_ids.intersection(measurement['identity']), 'Physical trace counted twice')
        physical_ids.update(measurement['identity'])
        case=cases[ident]; key=case['contract'],tuple(case['shape'])
        endpoints=groups.setdefault(key,{})
        require(case['chip'] not in endpoints,'Unexpected duplicate SKU')
        row={'case_id':ident,'ca_us':measurement['ca_us'],'aic':phases['active_mmad_aics'],
             'producer_aiv':phases['active_vector_instruction_aivs'],'measurement':measurement,
             'case':case}
        for version in VERSIONS:
            target=plans[version][ident]['target']
            future=plans[version][ident]['970dt']
            target_case=by_id[version][ident]
            validate_schedule(target,frozen[version]['chips'][case['chip']])
            validate_schedule(future,frozen[version]['chips']['Ascend970DT'])
            require(target['us']==target_case['predicted_us'] and future['us']==target_case['extrapolated_970dt_us']
                    and future['status']=='extrapolated', 'Frozen prices diverge')
            row[version]=target
            row[version+'_970dt']=future
        endpoints[case['chip']]=row
        observed[ident]=row
    require(seen==set(cases),'Missing complete predeclared cases')
    pairs=[]
    skus=SKUS
    for (contract,shape), by_chip in sorted(groups.items()):
        require(set(by_chip)==set(skus),'Paired DT and PR required')
        dt,pr=(by_chip[s] for s in skus)
        ratio=dt['ca_us']/pr['ca_us']
        pair={'contract':contract,'shape':shape,'id':declared[shape],'ca_dt_over_pr':ratio}
        for version in VERSIONS:
            pred=dt[version]['us']/pr[version]['us']
            pair[version]=dict(dt_over_pr=pred,absolute_relative_error_pct=100*abs(pred/ratio-1))
        pairs.append(pair)
    summary={}
    for contract in sorted({key[0] for key in groups}):
        summary[contract]={}
        for version in VERSIONS:
            by_sku={}
            for sku in skus:
                rows=[v[sku] for key,v in groups.items() if key[0]==contract]
                diffs=[abs(r[version]['us']-r['ca_us']) for r in rows]
                by_sku[sku]=dict(n=len(rows),wape=100*sum(diffs)/sum(r['ca_us'] for r in rows),
                                 maximum=100*max(d/r['ca_us'] for d,r in zip(diffs,rows)),
                                 aic_exact=sum(r['aic']==r[version]['aic'] for r in rows),
                                 aic_mae=mean(abs(r['aic']-r[version]['aic']) for r in rows),
                                 producer_aiv_exact=sum(r['producer_aiv']==r[version]['producer_aiv'] for r in rows))
            matching=[p for p in pairs if p['contract']==contract]
            errors=[p[version]['absolute_relative_error_pct'] for p in matching]
            summary[contract][version]=dict(sku=by_sku,paired_ratio_mape=mean(errors),
                                             paired_ratio_max=max(errors))
    return dict(summary=summary,pairs=pairs),observed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',nargs='?',type=Path,default=DEFAULT_ARCHIVE)
    parser.add_argument('--pairs',action='store_true')
    args=parser.parse_args()
    result,_=compare(load_comparison(args.archive))
    print(json.dumps(result['pairs' if args.pairs else 'summary'],indent=2))
