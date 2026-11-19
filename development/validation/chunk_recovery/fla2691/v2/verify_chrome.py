"""Reconcile Chrome B/E synchronization descriptors with actual per-core instruction shards.

msopprof marks synchronization intervals as excluded from instruction duration.
The merged view can replace their start by a dependency's start, retaining the
physical instruction PC, retirement and direction. Nonsync instructions retain
both start and end ticks. Check all records, all cores and every official duration.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from probe_v2 import probe,OUT,RAW,LOG,REG
from score import TICKS,tables,checked

def tick(ts):
    return round(ts*TICKS)

def content(path, merged):
    events=json.loads(path.read_bytes())['traceEvents']
    records=defaultdict(lambda:dict(by_instr=Counter(),sync=Counter(),first=10**20,last=-1,
                                    ends=[],labels=[],mmads=0,be_count=0,wait_spans=[]))
    open_pairs={}
    for event in events:
        phase=event['ph']
        if phase not in ('B','E','X'):continue
        core=event['pid'] if merged else path.parent.name
        record=records[core]
        if phase=='B':
            key=(core,event['tid'],event['id'],event['name'])
            checked(key not in open_pairs,'duplicate open B/E id')
            open_pairs[key]=event
            continue
        if phase=='E':
            key=(core,event['tid'],event['id'],event['name'])
            begin=open_pairs.pop(key)
            lo=tick(begin['ts']);hi=tick(event['ts'])
            event=begin;record['be_count']+=1
        else:
            lo=tick(event['ts']);hi=tick(event['ts']+event['dur'])
        checked(hi>=lo,'invalid instruction window')
        record['first']=min(lo,record['first']);record['last']=max(hi,record['last'])
        detail=json.loads(event['args']['detail'] or '{}')
        name=event['name'];pc=event['args']['pc_addr']
        if detail.get('is_sync') or detail.get('is_wait'):
            # Some merged WAIT_FLAG spans start at the predecessor's dispatch.
            # The retired instruction and its source/destination/ID must still match.
            signature=(name,pc,hi,detail.get('flag_id'),detail.get('sync_id'),detail.get('target_pipe'),detail.get('consumer_pipe'))
            record['sync'][signature]+=1
            if name=='WAIT_INTRA_BLOCK':
                record['wait_spans'].append(dict(pc=pc,start_tick=lo,end_tick=hi,display_id=event.get('id')))
        else:
            record['by_instr'][(name,pc,lo,hi)]+=1
        if name=='END':record['ends'].append([pc,lo,hi])
        if name=='END_LABEL':record['labels'].append([pc,lo,hi])
        if name=='MMAD':record['mmads']+=1
    checked(not open_pairs,'unmatched begin/end marker')
    return records

def multiset_digest(records):
    h=hashlib.sha256()
    for signature,count in sorted(records.items(),key=lambda kv:repr(kv[0])):
        h.update((json.dumps([signature,count],separators=(',',':'))+'\n').encode())
    return h.hexdigest()

def main():
    initial=OUT/'receipt.json';prior=json.loads(initial.read_bytes());plan=json.loads(REG.read_bytes())
    checked(prior['runtime_verified'] and prior['stages_verified'] and prior['correct_host']
            and prior['golden_qualified'] and prior['process_complete'] and prior['sync_issue_count']==0,
            'host, SKU, stages, correctness, process or synchronization')
    official=tables(LOG.read_text());checked(len(official)==len(prior['actual_launches'])==3,'unexpected stages')
    rows=[]
    for launch,durations in zip(prior['actual_launches'],official):
        sim=(probe.ROOT/launch['captured_binary']).parents[1]/'simulator'
        agg=sim/'trace.json';merged=content(agg,True)
        shards={p.parent.name:p for p in sim.glob('core*/trace.json')}
        checked(set(shards)==set(merged)==set(durations) and bool(shards),'core roster')
        details={}
        for core,rec in merged.items():
            src=content(shards[core],False)[core]
            for k in ('by_instr','sync','first','last','ends','labels','mmads'):
                checked(src[k]==rec[k],f'physical content mismatch in {k} for {core}')
            checked(len(src['ends'])==1 and len(src['labels'])<=1,'actual END/label')
            if src['labels']:
                marker,=src['labels'];checked(int(marker[0],16)==0x11111111 and marker[2]==src['last'],'label at endpoint')
            duration=(src['last']-src['first'])/TICKS
            checked(abs(duration-float(durations[core][0]))<=.00501,'official core duration')
            details[core]=dict(first_tick=src['first'],last_tick=src['last'],real_end=src['ends'][0],ca_end_label=src['labels'],
                official_us=durations[core][0],reconstructed_us=duration,mmad_count=src['mmads'],
                physical_non_sync=sum(src['by_instr'].values()),sync_instruction_count=sum(src['sync'].values()),
                chrome_sync_pairs=rec['be_count'],non_sync_digest=multiset_digest(src['by_instr']),
                sync_digest=multiset_digest(src['sync']),per_core_trace_sha256=probe.digest(shards[core]))
            if not src['labels']:
                last=[w for w in rec['wait_spans'] if w['end_tick']==src['last']]
                if last:details[core]['terminal_wait']=last
        lo=min(v['first_tick'] for v in details.values());hi=max(v['last_tick'] for v in details.values())
        aic=sum('cubecore' in c for c in details)
        rows.append(dict(kernel=launch['kernel'],family=launch['release_family'],active_aic=aic,active_aiv=len(details)-aic,
                         mmads=sum(v['mmad_count'] for v in details.values()),first_tick=lo,last_tick=hi,
                         ca_us=(hi-lo)/TICKS,device_text_sha256=launch['device_text_sha256'],
                         published_names=launch['published_names'],aggregate=probe.rel(agg),
                         aggregate_sha256=probe.digest(agg),cores=details))
    checked(all(p['last_tick']<n['first_tick'] for p,n in zip(rows,rows[1:])),'launch order or overlapping serial stages')
    probe.put(OUT/'verified.json',dict(role=plan['role'],qualified=True,scope='direct V2 one development shape on 950DT',
        registration_sha256=probe.digest(REG),initial_receipt=probe.rel(initial),initial_receipt_sha256=probe.digest(initial),
        complete_golden=prior['actual_numeric_groups'],input_vk_unchanged=prior['initial_unchanged'],sku=probe.CHIP,
        stages=rows,serialized_device_us=sum(r['ca_us'] for r in rows),
        device_elapsed_first_to_last_us=(rows[-1]['last_tick']-rows[0]['first_tick'])/TICKS,
        host_process=prior['process'],runtime=prior['host_actual_bindings'],
        chronology='Uses the same simulator clock across launches; sum of measured per-launch all-core windows excludes inter-launch scheduler gaps.'))
    print([(s['family'],s['ca_us'],s['active_aic'],s['active_aiv'],s['mmads']) for s in rows])

if __name__=='__main__':main()
