"""Reconstruct the actual M→MTE1 flag state, comparing CA callbacks only afterwards."""
from collections import Counter,defaultdict
from pathlib import Path
import json,re,sys
from packages import ROOT,OUT,RAW,read,save,probe

def process(label,receipt):
    d=read(receipt)
    if d.get('skipped'):return None
    r,=[s for s in d['stages'] if s['kernel'].startswith('ChunkKdaFwd_')]
    basepath=ROOT/r['raw_path'];start=int((basepath/'dump/pc_start_addr.txt').read_text(),16)
    issues=Counter();examples={};cores={}
    for path in sorted((basepath/'simulator').glob('core*.cubecore*/trace.json')):
        events=[];mmads=0
        for index,e in enumerate(json.loads(path.read_bytes())['traceEvents']):
            if e['ph']!='X': continue
            if e['name']=='MMAD':mmads+=1
            if e['name'] not in ('SET_FLAG','WAIT_FLAG'):continue
            x=json.loads(e['args']['detail'])
            if x.get('target_pipe')!='CUBE' or x.get('consumer_pipe')!='MTE1':continue
            t=round(e['ts']*1650);pc=int(e['args']['pc_addr'],16)-start
            events.append((t,index,e['name'],x['flag_id'],pc))
        token={};per=Counter();prev=defaultdict(list);first_pending={};first_wait=set();prefix_proof={}
        for t,index,op,flag,pc in sorted(events):
            if op=='SET_FLAG':
                if flag not in first_wait and str(flag) not in prefix_proof:
                    if flag in first_pending:
                        early=first_pending[flag]
                        prefix_proof[str(flag)]=dict(first_set_tick=early[0],first_set_offset=hex(early[1]),
                            second_set_tick=t,second_set_offset=hex(pc))
                    else:first_pending[flag]=(t,pc)
                if flag in token:
                    issues[pc]+=1;per['set_while_already_set']+=1
                    if (pc,flag) not in examples:
                        examples[(pc,flag)]=dict(core=path.parent.name,flag=flag,
                          last_set_tick=token[flag][0],last_set_offset=hex(token[flag][1]),
                          second_set_tick=t,second_set_offset=hex(pc))
                token[flag]=(t,pc)
            else:
                first_wait.add(flag)
                if token.pop(flag,None) is None:per['wait_without_unconsumed_set']+=1
            per[op]+=1
            if len(prev[flag])<5:prev[flag].append(dict(name=op,tick=t,pc=hex(pc)))
        cores[path.parent.name]=dict(counts=per,mmad=mmads,end_pending={str(f):[t,hex(pc)] for f,(t,pc) in token.items()},
            initial_double_set_before_any_wait=prefix_proof,first_five_per_id=prev if path.parent.name in ('core0.cubecore0','core1.cubecore0') else {},source_sha256=probe.digest(path))
    callback=Counter({int(k,16)-start:n for k,n in d['sync_fault_pc_counts'].items()})
    return dict(label=label,release='26.10.0' if receipt.parent==OUT else '26.9.1',soc=d['case']['sku'],kernel=r['kernel'],
        device_text_sha256=r['device_text_sha256'],registration_sha256=d['registration_sha256'],receipt_sha256=probe.digest(receipt),
        input_shape=d['case']['shape'],metadata=d['case']['metadata'],cores=cores,unconsumed_flag_in_cores_at_end=sum(bool(v['end_pending']) for v in cores.values()),
        total_waits_without_token=sum(v['counts']['wait_without_unconsumed_set'] for v in cores.values()),
        boolean_replay_duplicate_sets=sum(issues.values()),reported_duplicate_sets=sum(callback.values()),
        kernel_init_prefix_certificates=sum(len(v['initial_double_set_before_any_wait']) for v in cores.values()),
        cores_with_all_3_initial_double_sets=sum(set(v['initial_double_set_before_any_wait'])=={'0','1','2'} for v in cores.values()),
        predicted_and_reported_counts_match_every_pc=issues==callback,
        readback_minus_reported={hex(k):issues[k]-callback[k] for k in set(issues)|set(callback) if issues[k]!=callback[k]},
        edges=list(examples.values()),state_definition='Each hardware token begins clear per AIC, source CUBE, destination MTE1, and event ID. On WAIT consume; on SET record excess if already live then set. Prefix certificates require two SETs of the same ID before the trace has issued any matching WAIT. Later boolean replay can differ from CA callback counts under overlapping/excluded waits; the callback count remains authoritative. Use only actual per-core X instructions sorted by issue timestamp and shard order at equal tick. Compare to independent simulator callback log, normalized by the captured ELF base.')
def main():
    old=ROOT/'results/v0.24/kda-v1';records=[]
    for prefix,folder in [('baseline',old),('release',OUT)]:
        for ident in ('dt-t2-h1-varlen','pr-t2-h1-varlen','dt-t64-h2-dense'):
            p=folder/(ident+'.json')
            if p.is_file():
                r=process(prefix+'-'+ident,p)
                if r:records.append(r)
    save(OUT/'m-mte1-retirement.json',dict(cases=records))
    print([(r['label'],r['kernel_init_prefix_certificates'],r['reported_duplicate_sets'],r['predicted_and_reported_counts_match_every_pc'],r['total_waits_without_token'],r['unconsumed_flag_in_cores_at_end']) for r in records])
if __name__=='__main__':main()
