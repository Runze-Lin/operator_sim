"""Read-only verification of finished per-launch device data after a profiler storage cap."""
from collections import Counter, defaultdict
from hashlib import blake2b, sha256
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/rms-dispatch-audit'
RAW = ROOT/'runs/rms-ordered-audit'
KERNEL = 'RmsNorm_normal_all_bf16_high_performance_5000'
VENDOR = ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/rms_norm/RmsNorm_normal_all_bf16_high_performance.o'
CLOCK=1650
MOD=1<<128


def checked(condition, reason):
    if not condition: raise ValueError(reason)


def rel(p): return str(p.relative_to(ROOT))


def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def text_bytes(p):
    d=p.read_bytes();checked(d[:6]==b'\x7fELF\x02\x01','ELF header')
    table=struct.unpack_from('<Q',d,40)[0];size,n,names=struct.unpack_from('<HHH',d,58)
    a,l=struct.unpack_from('<QQ',d,table+size*names+24);ss=d[a:a+l]
    for i in range(n):
        h=table+size*i;offset=struct.unpack_from('<I',d,h)[0]
        if ss[offset:].split(b'\0',1)[0]==b'.text':
            a,l=struct.unpack_from('<QQ',d,h+24);return d[a:a+l]
    raise ValueError('ELF missing text')


def events(path):
    """Strictly consume the complete, flat CANN traceEvents array in bounded memory."""
    decoder=json.JSONDecoder(); sep=False
    with path.open() as f:
        buf=f.read(1024*1024);match=re.match(r'^\s*\{\s*"displayTimeUnit"\s*:\s*"ns"\s*,\s*"profilingType"\s*:\s*"op"\s*,\s*"schemaVersion"\s*:\s*[12]\s*,\s*"traceEvents"\s*:\s*\[',buf)
        checked(match,'trace schema '+str(path));i=match.end()
        while True:
            if len(buf)-i<4096:
                buf=buf[i:]+f.read(1024*1024);i=0
            while i<len(buf) and buf[i].isspace():i+=1
            if buf[i:i+1]==']':
                checked((buf[i+1:]+f.read()).strip()=='}','incomplete/trailing JSON');break
            if sep:
                checked(buf[i:i+1]==',','missing trace event separator');i+=1
                while i<len(buf) and buf[i].isspace():i+=1
            try: value,i=decoder.raw_decode(buf,i)
            except json.JSONDecodeError:
                tail=buf[i:]; extra=f.read(1024*1024);checked(extra,'truncated JSON event')
                buf=tail+extra;value,i=decoder.raw_decode(buf,0)
            if value.get('ph')=='X':yield value
            sep=True


class Span:
    def __init__(self):
        self.first=10**20;self.last=-1;self.physical_last=-1;self.entry=[];self.end=[];self.label=[]
        self.events=0;self.content=0;self.writes=Counter();self.widths=set();self.yrows=0

    def add(self,x,base):
        pc=int(x['args']['pc_addr'],16);n=x['name'];a=round(x['ts']*CLOCK);b=round((x['ts']+x['dur'])*CLOCK)
        checked(b>=a,'negative duration');self.first=min(self.first,a);self.last=max(self.last,b)
        if n!='END_LABEL':self.physical_last=max(self.physical_last,b)
        if n=='END_LABEL':self.label.append((a,b,pc))
        if n=='END':self.end.append((a,b,pc-base))
        if pc==base+0x3e00 and n=='MOV_XD_SPR':self.entry.append(a)
        detail=x['args']['detail'];
        if n=='MOV_SRC_TO_DST_ALIGNv2' and '"instr_type":"MTE3"' in detail:
            d=json.loads(detail); key=pc-base; self.writes[key]+=d['processed_bytes']
            if key==0x5128:
                self.widths.add(d['burst_len']);self.yrows+=d['burst_number']
        # Sync flow annotations are represented differently by the merged trace.
        if '"is_sync":1' in detail or '"is_wait":1' in detail:return
        self.events+=1
        packet=f'{pc}:{n}:{a}:{b}'.encode();self.content=(self.content+int.from_bytes(blake2b(packet,digest_size=16).digest(),'big'))%MOD

    def signature(self):return (self.first,self.last,self.physical_last,self.entry,self.end,self.label,self.events,self.content,self.writes,self.widths,self.yrows)


def tables(log):
    matches=re.finditer(r'core_name\s+duration_time\(us\)\s+running_time\(us\)[^\n]*\n((?:core\d+\.[a-z]+\d+[^\n]*(?:\n|$))+)',log)
    blocks=[]
    for m in matches:
        rows={}
        for l in m[1].splitlines():
            core,d,r=l.split();checked(core not in rows,'repeated core in duration table');rows[core]=(d,r)
        blocks.append(rows)
    return blocks


def get_host(folder,chip,plan):
    log=(folder/'run.log').read_text();inputs=(folder/'inputs.txt').read_text().splitlines()
    checked(re.findall(r'^runtime_soc=(\S+)',log,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if 'PR' in chip else 'hbm3_8000M.ini') in log,'actual runtime or physical DRAM')
    full_host = 'SMOKE_PASS' in log and 'SMOKE_FAIL' not in log and 'Profiling running finished. All task success.' in log
    checked(not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log),'CA error')
    spans=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',log,re.M|re.S)
    launches = re.findall('Start profiling on kernel: (\\S+)',log)
    checked(len(inputs)==len(plan)==10 and len(spans)<=len(launches)<=10 and all(k==KERNEL for k in launches),'input/batch/kernel count')
    if full_host:checked(len(spans)==len(launches)==10,'missing completed batch in full host')
    host=[]
    for job,(i,block),line in zip(plan,spans,inputs):
        i=int(i);r,w=job['shape'];checked(i==len(host)==job['i'] and line==f'rmsnorm {i} {r} {w} 0','input ordinal/shape')
        checked(f'case=rmsnorm api=aclnnRmsNorm R={r} C={w} dtype=BF16' in block and re.findall(r'Start profiling on kernel: (\S+)',block)==[KERNEL], 'actual host case')
        gold=re.findall(r'^correctness [^\n]+',block,re.M); counts=[int(re.match(r'correctness count=(\d+) bad=0\b',x)[1]) for x in gold]
        checked(counts==[r*w,r],'two golden groups')
        begins=re.findall(r'\[(\d+)\] \[TASK_BEGIN\].*stream_id:(\d+), task_id:(\d+) full_task_id:(\d+) BEGIN',block)
        dones=re.findall(r'\[(\d+)\] \[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE',block)
        begin,=set(begins);done,=set(dones)
        checked(len(begins)==len(dones) and begin[1:3]==done[1:3] and int(begin[3])==i and int(done[0])>=int(begin[0]),'task identity/termination')
        host.append(dict(golden=gold,begin_tick=int(begin[0]),done_tick=int(done[0]),full_task_id=int(begin[3]),raw_stream_id=int(begin[1])))
    return host,tables(log),full_host


def launch(d,chip,job,host,table,elf):
    i=job['i'];r,w=job['shape'];dump=d/'dump';binary=dump/'aicore_binary.o';base=int((dump/'pc_start_addr.txt').read_text(),16)
    checked(text_bytes(binary)==elf,'captured versus published .text')
    a,=re.findall(r'\s([a-f\d]{16})\s+\d+\s+FUNC\s+GLOBAL[^\n]+\s'+KERNEL+r'$',subprocess.check_output(['readelf','-Ws','--wide',str(binary)],text=True),re.M)
    checked(int(a,16)==0x3e00,'entry symbol');files=sorted((d/'simulator').glob('core*/trace.json'))
    checked(bool(table) and {p.parent.name for p in files}==set(table),'missing/extra core shards against official duration table')
    refs={};shas={};cores={};writes=Counter();nrows=0
    for p in files:
        c=p.parent.name;s=Span()
        for x in events(p):s.add(x,base)
        checked(s.entry==[s.first] and len(s.end)==len(s.label)==1 and s.end[0][2]==0x51d0 and s.label[0][1]==s.last and s.label[0][2]==0x11111111 and s.end[0][1]<=s.last,'missing ELF entry/physical END/terminal label on '+c)
        dur,runtime=table[c];duration=(s.last-s.first)/CLOCK
        checked(abs(duration-float(dur))<=.005+1e-5,'per-core duration disagrees on '+c+': '+str((duration,dur)))
        checked(s.widths=={2*w} and set(s.writes)=={0x4f38,0x5128} and s.writes[0x4f38]==4*s.yrows and s.writes[0x5128]==2*w*s.yrows,'per-core write stencil or width')
        refs[c]=s;shas[c]=digest(p);writes+=s.writes;nrows+=s.yrows
        cores[c]=dict(first_tick=s.first,last_tick=s.last,physical_last_tick=s.physical_last,physical_end_tick=s.end[0][0],official_duration_us=dur,official_running_us=runtime,
                      instruction_events_without_sync=s.events,signature128=f'{s.content:032x}',rows_written=s.yrows,shard_sha256=shas[c])
    checked(nrows==r and writes[0x4f38]==4*r and writes[0x5128]==2*r*w,'aggregate data plane shape mismatch')
    agg=d/'simulator/trace.json';merged=defaultdict(Span)
    for x in events(agg):merged[x['pid']].add(x,base)
    checked(merged.keys()==refs.keys(),'incomplete active core roster in aggregate')
    checked(all(refs[c].signature()==merged[c].signature() for c in refs),'instruction content mismatch between aggregate and per-core trace')
    first=min(x.first for x in refs.values());last=max(x.last for x in refs.values());physical=max(x.physical_last for x in refs.values())
    windows={c:[(s.first-first)/CLOCK,(s.last-first)/CLOCK] for c,s in refs.items()}
    stage=dict(kernel=KERNEL,ca_us=(last-first)/CLOCK,active_aiv=len(refs),active_aic=0,windows=windows,trace=rel(agg),trace_sha256=digest(agg),binary_sha256=digest(binary),
               first_tick=first,last_tick=last,physical_last_tick=physical,physical_ca_us=(physical-first)/CLOCK,cores=cores,ordinal=i)
    return dict(**job,timing_qualified=True,correctness=host['golden'],host_task={k:v for k,v in host.items() if k!='golden'},stage=stage,active_count_matches=len(refs)==job['active_aiv'])


def incomplete_evidence(d,table,reason):
    diag={'official_active_cores':len(table),'aggregate':rel(d/'simulator/trace.json')}
    match=re.search(r'on (core\d+\.veccore[01])$',reason)
    if not match:return diag
    core=match[1]; agg=d/'simulator/trace.json';sh=d/'simulator'/core/'trace.json'; diag['core']=core;diag['official_duration_us']=table[core][0]
    base=int((d/'dump/pc_start_addr.txt').read_text(),16)
    for role,path in [('shard',sh),('aggregate',agg)]:
        span=Span();counts=Counter()
        for x in events(path):
            if role=='aggregate' and x['pid']!=core:continue
            span.add(x,base)
            if x['name'] in ('DCI','END','END_LABEL'):counts[x['name']]+=1
        diag[role]=dict(path=rel(path),sha256=digest(path),entry_ticks=span.entry,last_seen_tick=span.last,terminal_opcode_counts=dict(counts))
    return diag


def main(chip):
    folder=RAW/chip;plan=[r for r in json.loads((OUT/'freeze.json').read_bytes())['rows'] if r['chip']==chip and r['protocol']=='ordered new sequence'];orig=json.loads((OUT/f'ordered-{chip}.json').read_bytes());cost=json.loads((folder/'cost.json').read_bytes());h,t,full_host=get_host(folder,chip,plan);elf=text_bytes(VENDOR)
    root,=folder.glob('profile/OPPROF_*/device0/'+KERNEL);checked(len(t)<=len(plan) and (not full_host or len(t)==len(h)==len(plan)), 'duration tables count');rows=[]
    for j in plan:
        i=j['i']
        if i>=min(len(h),len(t)):
            rows.append(dict(**j,timing_qualified=False,stage=None,correctness=h[i]['golden'] if i<len(h) else [],reason='no complete host case with paired golden and official duration table'));continue
        task,table=h[i],t[i]
        try:row=launch(root/str(j['i']),chip,j,task,table,elf)
        except (ValueError,IndexError,KeyError,TypeError,FileNotFoundError) as error:
            reason=str(error);row=dict(**j,timing_qualified=False,stage=None,correctness=task['golden'],reason=reason, incomplete_evidence=incomplete_evidence(root/str(j['i']),table,reason))
        print(chip,j['i'],row['timing_qualified'],row.get('reason',''),flush=True);rows.append(row)
    good=[r for r in rows if r['timing_qualified']];checked(all(r['stage']['first_tick']>p['stage']['last_tick'] for r,p in zip(good[1:],good)),'chronological order across invocations')
    data=dict(chip=chip,qualified=len(good)==len(rows),qualified_launches=len(good),rows=rows,original_process_qualified=orig['qualified'],original_process=cost,
              source_hashes={rel(p):digest(p) for p in [folder/'run.log',folder/'inputs.txt',OUT/'freeze.json',OUT/f'ordered-{chip}.json',VENDOR]},device_elf_text_sha256=sha256(elf).hexdigest(),official_tables=len(t),host_goldens=2*len(h),full_host_numeric_exit=full_host,
              scope='Per-launch full all-core device data; original profiler disposition is retained. All-core first-to-last span uses the CA endpoint label; physical opcode-only span is separately provided.')
    out=OUT/f'verified-ordered-{chip}.json';temp=out.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(out)

if __name__=='__main__':
    for chip in sys.argv[1:]:main(chip)
