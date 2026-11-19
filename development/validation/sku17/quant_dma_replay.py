"""Read only former (v0.16) PR holdout descriptors after the v0.17 freeze."""
import json
from pathlib import Path
from hashlib import sha256
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.quant_resource import resource_quant
from operator_model.chips import load_chip


def main():
    pr=load_chip('Ascend950PR_957b')
    base=ROOT/'results/v0.16'
    hold={r['id']:r for r in json.loads((base/'sku/holdout.json').read_text())}
    rows=[]
    for old in json.loads((base/'quant-pr-phases.json').read_text())['rows']:
        ref=hold[old['id']]['pr']
        path=ROOT/ref['trace']
        per=path.parent/old['critical_core']/'trace.json'
        # Per-core exports may have a shifted absolute origin. Only use
        # differences between points within the same export here.
        b=per.read_bytes()
        es=[e for e in json.loads(b)['traceEvents'] if e['ph']=='X' and e['name']=='MOV_SRC_TO_DST_ALIGNv2']
        writes=sorted((e for e in es if json.loads(e['args']['detail'])['instr_type']=='MTE3'),key=lambda e:e['ts'])[:2]
        w1,w2=writes
        observed=old['observed']
        win=max(e['ts']+e['dur'] for e in writes)-w1['ts']
        assert abs(win-observed['write'])<1e-5 and old['trace_sha256']==ref['trace_sha256']
        us,_,_,d=resource_quant(old['shape'],pr)
        first=d['first_tile_cycles']
        rows.append(dict(shape=old['shape'],ca_us=old['ca_us'],old_us=old['prediction_us'],new_us=us,
                    scale_granule_writers=d['scale_peak_writers'],observed=observed,old=old['predicted'],
                    new={k:first[k]/(1000*pr.frequency_ghz) for k in ('read_latency','write','vector')},
                    data_us=w1['dur'],scale_us=w2['dur'],issue_gap_us=w2['ts']-w1['ts'],
                    per_core_trace=str(per.relative_to(ROOT)),per_core_sha256=sha256(b).hexdigest()))
    def metric(field):return 100*sum(abs(r[field]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows)
    out=dict(role='old PR phase replay after freeze, excluded from new holdout',
             old_wape_pct=metric('old_us'),new_wape_pct=metric('new_us'),rows=rows)
    (ROOT/'results/v0.17/quant-pr-replay.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:
        ds=lambda key:[round(r[v][key],3) for v in ('old','new','observed')]
        print(r['shape'],r['scale_granule_writers'],ds('read_latency'),ds('write'),*[round(r[k],3) for k in ('data_us','scale_us','old_us','new_us','ca_us')])
    print(out['old_wape_pct'],out['new_wape_pct'])


if __name__=='__main__':
    main()
