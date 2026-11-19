"""Separate old development replays from post-freeze dual-SKU physical measurements."""
import json
import sys
from tempfile import NamedTemporaryFile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
OUT=ROOT/'results/v0.21/fusion'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
OPS=('attn_res_fwd','mhc_pre_sinkhorn','mhc_sinkhorn')
def read(p):return json.loads(p.read_text())
def write(p,d):
    with NamedTemporaryFile(mode='w', dir=p.parent, delete=False, suffix='.tmp') as f:
        json.dump(d, f, indent=2, allow_nan=False)
        f.write('\n')
        tmp=Path(f.name)
    tmp.replace(p)
def stats(rows,key):
    if not rows: return None
    return dict(n=len(rows),wape=sum(abs(r[key]-r['ca_us'])for r in rows)/sum(r['ca_us'] for r in rows),
                max_relative_error=max(abs(r[key]/r['ca_us']-1) for r in rows))
def report():
    dev=[]
    for p in read(ROOT/'results/v0.17/sku/pairs.json'):
        if p['op'] not in OPS:continue
        for side,chip in [('dt',DT),('pr',PR)]:
            z=p[side];r=estimate(p['op'],p['shape'],p['attrs'],chip=chip)
            dev.append(dict(op=p['op'],chip=chip,shape=p['shape'],attrs=p['attrs'],pred_us=r.latency_us,
                            active=[r.active_aic,r.active_aiv],old_us=p[side+'_us'],ca_us=p[side+'_ca_us'],
                            kernel=z['kernel'],ca_active=[z['active_aic'],z['active_aiv']],trace=z['trace_sha256']))
    for z in read(ROOT/'results/v0.19/workloads/attn-res-extension-evaluation.json')['rows']:
        s=z['shape'];a=dict(addend=True,materialize=True,write_idx=s[-1])
        r=estimate('attn_res_fwd',s,a,chip=z['chip'])
        dev.append(dict(op='attn_res_fwd',shape=s,attrs=a,chip=z['chip'],pred_us=r.latency_us,
                        old_us=z['pred_us'],ca_us=z['ca_us'],kernel=z['kernel'],trace=z['trace_sha256']))
    hold=[];failed=[];started=0
    reg=read(OUT/'freeze.json')
    forecasts={(r['op'],tuple(r['shape']),r['chip']):r for r in reg['cases']}
    for (op,s,chip),r in forecasts.items():
        f=OUT/'receipts'/('_'.join((op,*map(str,s),chip))+'.json')
        if not f.is_file():continue
        started+=1
        d=read(f)
        if not d['qualified']:
            failed.append(dict(op=op,shape=s,chip=chip,attempt='original',reason=d['reason'],cost=d['cost']))
            retry=OUT/'retries'/f.name
            if not retry.exists():continue
            d=read(retry)
            if not d['qualified']:
                failed.append(dict(op=op,shape=s,chip=chip,attempt='retry',reason=d['reason'],cost=d['cost']))
                continue
        assert d['cost']['started_utc']>reg['utc'] and d['shape']==list(s) and d['chip']==chip
        k=d['kernel']
        hold.append(dict(op=op,shape=s,attrs=r['attrs'],chip=chip,pred_us=r['latency_us'],old_us=r['v020_us'],
                         ca_us=k['ca_us'],future_us=forecasts[(op,s,FUT)]['latency_us'],query_ms=r['query_ms'],
                         active=[r['plan']['active_aic'],r['plan']['active_aiv']],ca_active=[k['active_aic'],k['active_aiv']],
                         kernel=k['kernel'],correctness=d['correctness'],mmad=k['mmad'],trace=k['trace'],
                         trace_sha256=k['trace_sha256'],cost=d['cost']))
    def summary(rows):
        return {op:{chip:{which:stats([r for r in rows if r['op']==op and r['chip']==chip],key)
                               for which,key in [('current','pred_us'),('v020','old_us')]} for chip in (DT,PR)}
                for op in OPS}
    write(OUT/'evaluation.json', dict(summary=summary(hold),development_summary=summary(dev),
          dev_reference='v0.17/sku/pairs and v0.19/workloads/attn-res-extension-evaluation (reused for calibration)',
          development_rows=dev,independent_rows=hold,rejections=failed,queued=60-started))
    for op in OPS:
        print(op,summary(hold)[op],summary(dev)[op])
if __name__=='__main__':report()
