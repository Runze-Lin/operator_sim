"""Verify callback PCs in each captured ELF and find strong intra-launch flag evidence."""
import argparse
from collections import Counter
import hashlib,json,re,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];DEST=ROOT/'results/v0.27/kda/host-control';RAW=ROOT/'results/v0.27/kda/raw/host-control'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kda_v1_24'))
from inspect_sync import code_map

def read(p):return json.loads(p.read_bytes())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prefixes(shard,base,callbacks):
    ordered=[]
    for seq,e in enumerate(read(shard)['traceEvents']):
        if e.get('ph')!='X' or e['name'] not in {'SET_FLAG','WAIT_FLAG'}:continue
        d=json.loads(e['args']['detail'])
        if d.get('target_pipe')=='CUBE' and d.get('consumer_pipe')=='MTE1':
            ordered.append((e['ts'],seq,e['name'],d['flag_id'],int(e['args']['pc_addr'],16)-base))
    first={};wait=set();proof={}
    for ts,seq,op,flag,pc in sorted(ordered):
        tick=round(ts*1650)
        if op=='WAIT_FLAG':wait.add(flag)
        elif flag not in wait and flag not in proof:
            if flag in first:proof[flag]=dict(first_tick=first[flag][0],first_offset=hex(first[flag][1]),second_tick=tick,second_offset=hex(pc),second_pc_in_actual_callback=pc in callbacks)
            else:first[flag]=(tick,pc)
    return dict(source_sha256=digest(shard),evidence=proof)

def make():
    if not (DEST/'dt.json').exists():return False
    receipt=read(DEST/'dt.json');launch=[];current=None;counts=Counter()
    for line in (RAW/'run.log').read_text(errors='replace').splitlines():
        m=re.search(r'Start profiling on kernel: (\S+)',line)
        if m:
            name=m.group(1);current=Counter();launch.append((name,counts[name],current));counts[name]+=1
        m=re.search(r'execute_set_flag already has same set_flag! pc:(0x[\da-fA-F]+)',line)
        if m and current is not None:current[int(m.group(1),16)]+=1
    assert [(n,i) for n,i,c in launch]==[(s['kernel'],s['ordinal']) for s in receipt['stages']]
    prev=read(ROOT/'results/v0.24/kda-v1/dt-t2-h1-varlen-sync.json');earlier={int(x['offset'],16) for x in prev['faults']};rows=[]
    for (n,i,c),s in zip(launch,receipt['stages']):
        if not n.startswith('ChunkKdaFwd_'):continue
        p=ROOT/s['raw_path'];binary=p/'dump/aicore_binary.o';base=int((p/'dump/pc_start_addr.txt').read_text(),16)
        rel={pc-base:num for pc,num in c.items()};ops=code_map(binary,set(rel));assert set(ops)==set(rel)
        cores={f.parent.name:prefixes(f,base,set(rel)) for f in sorted((p/'simulator').glob('core*.cubecore*/trace.json'))}
        rows.append(dict(physical_launch=i,source_stage_order=['gate_prepare','post_wu','fwd_h','finalize'][len(rows)],device_text_sha256=s['device_text_sha256'],same_text_as_one_sequence_control=s['device_text_sha256']==prev['captured_text_sha256'],
          captured_binary_sha256=digest(binary),pc_base=hex(base),duplicate_callbacks=sum(rel.values()),unique_offsets=len(rel),common_sites_with_single_sequence=len(set(rel)&earlier),opcode_counts=dict(Counter({mn:sum(rel[x] for x in rel if ops[x]['mnemonic']==mn) for mn in {ops[x]['mnemonic'] for x in rel}})),
          faults=[dict(relative=hex(k),count=rel[k],**ops[k]) for k in sorted(rel)],
          aic_count=len(cores),aic_with_any_two_sets_before_first_matching_wait=sum(bool(x['evidence']) for x in cores.values()),initial_event_certificates=sum(len(x['evidence']) for x in cores.values()),cores=cores))
    info=dict(source_receipt_sha256=digest(DEST/'dt.json'),source_log_sha256=digest(RAW/'run.log'),original_single_sequence_direct_control=str((ROOT/'results/v0.24/kda-v1/dt-t2-h1-varlen-sync.json').relative_to(ROOT)),
              scope='physical ascending ordinals within the same fused family. Stage names derive from the pinned host stage loop. Tick origins are per simulator run. A missing prefix certificate cannot rule out a later repeated flag.',phases=rows)
    out=DEST/'fault-sites.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(info,indent=2)+'\n');tmp.replace(out);return True
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--follow',action='store_true');a=p.parse_args();
    while not make() and a.follow:time.sleep(20)
