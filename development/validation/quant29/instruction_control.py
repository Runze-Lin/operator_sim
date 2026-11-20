"""Two archived, official [2048,5120] controls check the same END/roster/MTE3 assumptions."""
from collections import Counter
import json
import re
from study import DT,PR,OUT,ROOT,FROZEN,read,store
from development.validation.rms_ordered_verify.verify import digest,events,tables,text_bytes


def run():
    reg=read(FROZEN)
    rows=read(ROOT/'results/v0.19/sku/records.json')
    result=[]
    for sku in (DT,PR):
        v=next(v for v in rows if v['op']=='quant' and v['chip']==sku and v['shape']==[2048,5120])
        tr=ROOT/v['trace'][0]
        assert digest(tr)==v['trace_sha256'][0]
        assert text_bytes(tr.parents[1]/'dump/aicore_binary.o')==text_bytes(ROOT/reg['official']['vendor'])
        log=(ROOT/v['run_log']).read_text(errors='replace');tab,=tables(log)
        n=2048*5120
        assert re.findall(r'^correctness quant_elements=(\d+) scale_elements=(\d+) bad=(\d+) comparison=exact_bytes',log,re.M)==[(str(n),str(n//32),'0')]
        bounds={};ends=Counter();labels={};writes=Counter();reads=Counter();vector=Counter()
        for e in events(tr):
            c=e['pid'];name=e['name'];start=round(1650*e['ts']);finish=round(1650*(e['ts']+e['dur']))
            a=bounds.setdefault(c,[start,finish]);a[0]=min(a[0],start);a[1]=max(a[1],finish)
            ends[c]+=name=='END';vector[c]+=name=='VF'
            if name=='END_LABEL':
                assert c not in labels
                labels[c]=finish
            if name.startswith('MOV_SRC_TO_DST'):
                d=json.loads(e['args']['detail'])
                if not (d['is_wait'] or d['is_sync'] or d['is_duration_excluded']):
                    if d['queue_type']=='MTE3':writes[c]+=d['processed_bytes']
                    if d['queue_type']=='MTE2':reads[c]+=d['processed_bytes']
        assert set(bounds)==set(tab)==set(writes)
        assert set(ends.values())=={1} and all(labels.get(c,b)==b and abs((b-a)/1650-float(tab[c][0]))<.025 for c,(a,b) in bounds.items())
        us=(max(b for a,b in bounds.values())-min(a for a,b in bounds.values()))/1650
        assert abs(v['ca_us']-us)<.002 and sum(writes.values())==n+n//32
        def vals(d):return dict(min=min(d.values()),max=max(d.values()),total=sum(d.values()))
        result.append(dict(sku=sku,shape=v['shape'],trace=v['trace'][0],source='results/v0.19/sku/records.json',trace_sha256=v['trace_sha256'][0],qualified_instruction_us=us,original_us=v['ca_us'],active_aiv=len(bounds),terminal_counts=dict(Counter(ends.values())),read_bytes_per_core=vals(reads),write_bytes_per_core=vals(writes),vf_calls_per_core=vals(vector),
            note='Earlier three_ops host used one combined exact-output-and-scale golden line. Validated it in its original syntax. New large_vector host reports two count/bad lines.'))
    store(OUT/'instruction-controls.json',result)


if __name__=='__main__':run()
