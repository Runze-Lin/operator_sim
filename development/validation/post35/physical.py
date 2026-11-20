"""Reconstruct actual critical-lane DMA issue, response and drain from existing CA shards."""
from collections import Counter
from pathlib import Path
from statistics import median
import json,sys
from registration import D,P,OUT,ROOT,save,read,sha,rel
sys.path.insert(0,str(ROOT/'development/validation/coverage33'))
from rms_archive import moves,endpoint


def extract(receipt,case):
    chip=case['chip'];s=case['stage'];shape=case['shape'];h=shape[-1]
    require=case['timing_qualified'] and s['kernel'].endswith('_6')
    assert require and s['active_aiv']>0 and s['writes_bytes']==shape[0]*4*h*2
    core=max(s['windows'],key=lambda k:s['windows'][k][1])
    raw=ROOT/s['trace'];shard=raw.parent/core/'trace.json';ends=endpoint(shard)
    assert abs(max(e['retire'] for e in ends.values())-s['last_tick'])<2
    first=s['first_tick'];reads=[];writes=[]
    for e in moves(shard):
        det=json.loads(e['args']['detail'])
        if any(det.get(k) for k in ('is_wait','is_sync','is_duration_excluded')):continue
        x=dict(a=round(e['ts']*1650)-first,b=round((e['ts']+e['dur'])*1650)-first,
               bytes=det['processed_bytes'],ub=det['dst_addr'] if det['queue_type']=='MTE2' else det['src_addr'],
               gm=det['src_addr'] if det['queue_type']=='MTE2' else det['dst_addr'])
        if det['queue_type']=='MTE2':reads.append(x)
        elif det['queue_type']=='MTE3':writes.append(x)
    reads.sort(key=lambda x:x['a']);writes.sort(key=lambda x:x['a'])
    assert reads[0]['bytes']==96
    fs=2 if h==5120 else 1  # evidence extractor verifies both bytes and the alternating addresses below
    rows=len(writes)//fs;w=h//fs
    assert len(writes)==fs*rows and 1<=rows<=case['plan']['rows_per_core']
    assert [x['bytes'] for x in reads[1:]]==([64,16]+[2*w,8*w]*fs)*rows
    assert all(x['bytes']==8*w for x in writes)
    operand=reads[1:]; per=2+2*fs
    fr=[]
    for i in range(rows):
        co=operand[i*per:i*per+2]; pair=operand[i*per+2:(i+1)*per]
        for k in range(fs):
            hr,x=pair[2*k:2*k+2];wr=writes[i*fs+k]
            fr.append(dict(row=i,fragment=k,h=hr,x=x,coefficient=co if k==0 else None,write=wr,
                           ready=max(t['b'] for t in (*co,hr,x)),vf_handoff_ticks=wr['a']-max(t['b'] for t in (*co,hr,x))))
    if fs==2 and rows>1:
        assert fr[0]['h']['ub']==fr[2]['h']['ub'] and fr[0]['h']['ub']!=fr[1]['h']['ub']
        assert fr[0]['x']['ub']==fr[2]['x']['ub'] and fr[0]['x']['ub']!=fr[1]['x']['ub']
    gaps=[wr['a']-prev['a'] for prev,wr in zip(writes,writes[1:])]
    total=round(s['ca_us']*1650)
    phases=dict(first_write=writes[0]['a'],subsequent_write_issues=sum(gaps),
                last_write_response=writes[-1]['b']-writes[-1]['a'],exit=total-writes[-1]['b'])
    assert sum(phases.values())==total
    return dict(shape=shape,chip=chip,core=core,roles=dict(prior=case['mode'],collection=case['role']),rows=rows,
           first_write_tick=writes[0]['a'],write_intervals=gaps,read_starts=[x['h']['a'] for x in fr],
           slot_lead_to_store_ticks=[fr[k-2]['write']['a']-f['h']['a'] for k,f in enumerate(fr) if k>=2],
           second_row_reads_before_previous_store_done=(fr[fs]['h']['a']<fr[fs-1]['write']['b']) if rows>1 else None,
           actual_phases_ticks=phases,fragments=fr,ca_us=s['ca_us'],merged_sha256=s['trace_sha256'],shard=rel(shard),
           shard_sha256=sha(shard),original_receipt=rel(receipt),original_receipt_sha256=sha(receipt),all_core_retired=True)

def run():
    rows=[]
    for chip in (D,P):
        for group in ('width','many','steady','boundary'):
            f=ROOT/'results/v0.25/post/receipts'/f'{group}-{chip}.json'
            for c in read(f)['rows']:
                if not c['timing_qualified']:continue
                if c['shape'][::2] not in ([65,320],[65,5120],[449,512],[193,896],[181,1024],[57,192]):continue
                rows.append(extract(f,c))
    save(OUT/'physical.json',dict(rows=rows,source='Installed 9.2.1 four-stream BF16 aclnnMhcPost, token kernel suffix 6, nonwait packet issue/retire. All times in 950 ticks from system first instruction; merged fingerprints from the original qualified numeric/retirement receipt; this extraction hashes the actual critical physical shard.'))
if __name__=='__main__':run()
