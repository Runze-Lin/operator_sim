"""Rebuild the system-operator additions from checked numerical and instruction receipts."""
import argparse
from collections import defaultdict
import gzip
import json

from .__main__ import HERE, digest, encoded, ingest, metrics, read, require, sha_field, unseal
from .report_pages import DT, PR
from .routed_gmm import bundle as routed
from .quant_update import bundle as quant

SCOPES=('quant_mxfp4','quant_token_fp8','cache_load_mxfp4_antimx','routed_grouped_matmul_mxfp4_v5')
ARCHIVES={
    'quant_token_fp8':('token_quant/development.json.gz','token_quant/independent.json.gz'),
    'cache_load_mxfp4_antimx':('cache_load/fp4-development.json.gz','cache_load/fp4-independent.json.gz'),
}


def archive(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def phase_rows(path, contract, development):
    packed=archive(path)
    frozen=unseal(packed['freeze'],'freeze')
    cases={c['id']:c for c in frozen['cases']}
    for item in packed['observations']:
        receipt=item['receipt']
        obs=ingest(frozen,receipt)
        require(obs==item['measurement'] and obs['data_role']==('replay' if development else 'prospective'),
                f'Measurement role or receipt mismatch in {path}')
        c=cases.pop(obs['case_id'])
        ep=c['shape']
        if contract=='quant_token_fp8':
            require(c['api']=='aclnnDynamicQuantV2' and c['outputs']==dict(Y=ep[0]*ep[1],scale=ep[0])
                    and c['dtype']=='bf16' and not c['attrs'],'Invalid token quant outputs')
        else:
            from math import prod
            require(c['api']=='aclnnAntiMxQuant' and c['outputs']==dict(Y=prod(ep))
                    and c['dtype']=='mxfp4' and c['attrs']=={'implementation':'aclnnAntiMxQuant'},'Invalid CacheLoad')
        stage,=receipt['stages']
        hashes=stage['physical_shard_sha256']
        require(c['contract']==contract and set(hashes)==set(stage['launched_cores'])
                and all(sha_field(h) for h in hashes.values()) and digest(encoded(hashes))==stage['trace_sha256']
                and sha_field(item['probe_sha256']) and item['wall_seconds']>0,'Invalid physical evidence')
        row={k:v for k,v in c.items() if k not in ('outputs','id')}
        row.update({k:obs[k] for k in ('case_id','freeze_sha256','receipt_sha256','ca_us','kernel','provider','identity')})
        yield {**row,'data_role':'replay' if development else 'independent','query_970_ms':item['query_970_ms'],
               'acquisition_seconds':item['wall_seconds'],'signed_error_pct':100*(row['predicted_us']/row['ca_us']-1)}
    require(not cases,f'Missing qualified observations in {path}')


def campaign_rows():
    q=quant.load()
    for development in (True,False):
        yield from quant.rows(q,development=development)
    for name,(dev,hold) in ARCHIVES.items():
        yield from phase_rows(HERE/dev,name,True)
        yield from phase_rows(HERE/hold,name,False)
    # A frozen histogram at an already-seen (R,N,K) is a separate routing
    # experiment. Keep its receipt in the campaign, outside the new-shape score.
    native=list(routed.rows(routed.read(routed.DEFAULT)))
    seen={tuple(r['shape']) for r in native if r['data_role']=='replay'}
    yield from (r for r in native if r['data_role']=='replay' or tuple(r['shape']) not in seen)


def check_origin(rows):
    published={r['case_id']:r for r in rows if r['contract'] in SCOPES}
    seen=set()
    geometries=defaultdict(set)
    for r in campaign_rows():
        key=r['case_id']
        require(r==published.get(key) and key not in seen,'Reference differs from receipt and freeze: '+key)
        seen.add(key)
        geometries[r['contract'],r['data_role']].add(tuple(r['shape']))
    require(seen==set(published),'Unbound system operator window')
    for name in SCOPES:
        require(geometries[name,'replay'].isdisjoint(geometries[name,'independent']),'Reused development geometry')


def statistics(rows):
    m=metrics(rows)
    differences=[r['predicted_us']-r['ca_us'] for r in rows]
    return dict(n=m['windows'],shapes=m['shapes'],wape=m['wape_pct'],maximum=m['max_pct'],
                bias=100*sum(differences)/sum(r['ca_us'] for r in rows) if rows else None,
                faster_fraction=sum(d<0 for d in differences)/len(rows) if rows else None)


def refresh():
    folder=HERE/'reference'
    source=folder/'windows.jsonl'
    old=[s for s in source.read_text().splitlines(keepends=True) if json.loads(s)['contract'] not in SCOPES]
    new=list(campaign_rows())
    all_rows=[json.loads(s) for s in old]+new
    check_origin(all_rows)
    source.write_text(''.join(old)+''.join(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n' for r in new))
    summary=read(folder/'summary.json')
    for name in SCOPES:
        by_chip={}
        for chip in (DT,PR):
            group=[r for r in new if r['contract']==name and r['chip']==chip]
            by_chip[chip]={role:statistics(group if role=='all' else [r for r in group if r['data_role']==role])
                           for role in ('all','replay','independent')}
        summary[name]=by_chip
    def save(filename,d):
        (folder/filename).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    save('summary.json',summary)
    manifest=read(folder/'manifest.json')
    manifest.update(contracts=len(summary),protocol_windows=len(all_rows),
                    distinct_chip_and_stage_signatures=len({(r['chip'],tuple(r['identity'])) for r in all_rows}))
    manifest['scope']='Modeled contracts and complete 950 CA windows. Additional measured-only diagnostics are listed separately. 970DT is an extrapolation.'
    manifest['pricing_by_scope'].update(dict.fromkeys(SCOPES,'1.6.0'))
    files=['summary.json','windows.jsonl','../quant_update/calibration.json.gz','../routed_gmm/reference.json.gz',
           '../cache_load/fp8-smoke.json.gz','../elementwise/reference.json.gz','../quant_update/strided-950dt.json',
           '../quant_update/strided-950pr.json']+['../'+s for pair in ARCHIVES.values() for s in pair]
    for entry in files:
        manifest['export_sha256'][entry]=digest((folder/entry).read_bytes())
    save('manifest.json',manifest)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--refresh',action='store_true',required=True)
    cli.parse_args()
    refresh()
