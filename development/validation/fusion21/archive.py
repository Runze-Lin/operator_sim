"""Audit available reference windows across historical collection contexts, keyed by trace digest."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
OUT=ROOT/'results/v0.21/fusion'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
OPS=('attn_res_fwd','mhc_sinkhorn','mhc_pre_sinkhorn')

def data(p):return json.loads((ROOT/p).read_text())
def audit():
    samples={}
    def add(op,shape,a,chip,ca,digest,kernel,golden,source):
        if op not in OPS:return
        digs=[digest] if isinstance(digest,str) else digest
        assert digs and all(len(h)==64 for h in digs) and ca>0 and golden
        key=chip+':'+','.join(digs)
        if key in samples:
            assert samples[key]['shape']==list(shape) and abs(ca-samples[key]['ca_us'])<.004
            return
        est=estimate(op,shape,a,chip=chip)
        samples[key]=dict(op=op,shape=shape,attrs=a,chip=chip,ca_us=ca,pred_us=est.latency_us,
                          kernel=kernel,source=source,trace_sha256=digs,correctness=golden)
    source='results/v0.12/evaluation.json'
    for r in data(source)['rows']:
        if r['op'] in OPS:
            digest=r.get('db_sha256')
            if not digest:
                evidence=data(r['source'])
                digest=[k['db_sha256'] for k in evidence['measurement']['kernels'] if k.get('db_sha256')]
            add(r['op'],r['shape'],r['attrs'],DT,r['reference_us'],digest,r['kernel'],r['correctness'],source)
    source='results/v0.17/sku/records.json'
    for r in data(source):
        if r['op'] in OPS:
            add(r['op'],r['shape'],r['attrs'],r['chip'],r['ca_us'],r['trace_sha256'],r['kernel'],r['correctness'],source)
    for prefix in ('ca','holdout','micro-ca','extension-holdout'):
        source=f'results/v0.19/workloads/attn-res-{prefix}.json'
        for r in data(source):
            if r['qualified']:
                s=r['shape'];k=r['kernel']
                add('attn_res_fwd',s,dict(addend=True,materialize=True,write_idx=s[-1]),r['chip'],
                    k['ca_us'],k['trace_sha256'],k['kernel'],r['correctness'],source)
    summary=[]
    for op in OPS:
        for chip in (DT,PR):
            rows=[r for r in samples.values() if (r['op'],r['chip'])==(op,chip) and r['pred_us'] is not None]
            worst=max(rows,key=lambda r:abs(r['pred_us']/r['ca_us']-1))
            summary.append(dict(op=op,chip=chip,distinct_windows=len(rows),distinct_shapes=len({tuple(r['shape']) for r in rows}),
                wape=sum(abs(r['pred_us']-r['ca_us']) for r in rows)/sum(r['ca_us']for r in rows),
                max_relative_error=abs(worst['pred_us']/worst['ca_us']-1),worst_shape=worst['shape'],worst_attrs=worst['attrs']))
    (OUT/'historical-windows.json').write_text(json.dumps(dict(
        scope='Retrospective sensitivity including reused original development windows, earlier heldout windows and distinct direct controls. Mixed launch contexts, deduplicated by trace SHA; not an independent accuracy test.',
        summary=summary,rows=list(samples.values())),indent=2)+'\n')
    for r in summary:print(r['op'],r['chip'],r['distinct_windows'],round(100*r['wape'],2),round(100*r['max_relative_error'],2),r['worst_shape'])
if __name__=='__main__':audit()
