"""Rebuild one instruction-level recurrent state example from the archived 950PR trace."""
from collections import defaultdict
from hashlib import sha256
from math import isclose
from pathlib import Path
import json
import re
import sys
import struct

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate, load_chip

CHIP = 'Ascend950PR_957b'
RUNS = ROOT / 'runs/sku17/extension'
OUT = ROOT / 'results/v0.21/recurrent-example'
FIG = ROOT / 'docs/examples/figures/kda-recurrent-state-pr.png'
REDUCE = {0x7c34:'decay_key', 0x7d50:'update_query'}
PATTERN = re.compile(r'^\[info\] \[(\d+)\] \(PC: (0x[\da-f]+)\).*?\(ID: (\d+)\) (\w+)')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def elf_text(path):
    data = path.read_bytes()
    assert data[:6]==b'\x7fELF\x02\x01'
    table = struct.unpack_from('<Q', data, 40)[0]
    size,count,names = struct.unpack_from('<HHH', data, 58)
    off,n = struct.unpack_from('<QQ', data, table+names*size+24)
    strings=data[off:off+n]
    for i in range(count):
        h=table+i*size; pos=struct.unpack_from('<I',data,h)[0]
        if strings[pos:].split(b'\x00',1)[0]==b'.text':
            off,n=struct.unpack_from('<QQ',data,h+24);return data[off:off+n]
    raise ValueError('missing .text')


def read_receipt(tokens):
    prefix = f'kda_recurrent_1_{tokens}_1_1_128_128_{CHIP}'
    matches = []
    for path in sorted(RUNS.glob(prefix+'*/strict.json')):
        entry, = json.loads(path.read_bytes())
        log = (path.parent/'run.log').read_text()
        if entry['cost']['exit_code']==0 and 'SMOKE_PASS' in log and len(entry['correctness'])==3 and all('bad=0 ' in line for line in entry['correctness']):
            matches.append((path,entry))
    path, entry = matches[-1]
    assert entry['shape']==[1,tokens,1,1,128,128] and entry['chip']==CHIP and entry['active_aiv']==1 and entry['active_aic']==0
    return path,entry


def parse_dump(path):
    for line in path.open():
        m = PATTERN.match(line)
        if m:
            yield int(m[1]),int(m[2],16),int(m[3]),m[4]


def extract():
    preds = []
    for t in (1,2,3):
        qfile,q = read_receipt(t)
        e = estimate('kda_recurrent',q['shape'],q['attrs'],chip=CHIP,dtype='BF16')
        assert isclose(sum(e.projection['stages_us'].values()),e.latency_us,abs_tol=1e-8)
        preds.append(dict(tokens=t,model_us=e.latency_us,ca_us=q['ca_us'],exposed_us=e.projection['stages_us'],
                          qualified=str(qfile.relative_to(ROOT)),qualified_sha256=digest(qfile), golden=q['correctness']))
    stage, = q['stages']; trace=ROOT/stage['trace']; shard=trace.parent/'core0.veccore0/trace.json'
    dump, = qfile.parent.glob('profile/*/dump'); popped=dump/'core0.veccore0.instr_popped_log.dump'; retired=dump/'core0.veccore0.instr_log.dump'
    assert digest(trace)==stage['trace_sha256'] and digest(dump/'aicore_binary.o')==stage['binary_sha256']
    clock=1000*load_chip(CHIP).frequency_ghz; base=int((dump/'pc_start_addr.txt').read_text(),16)
    start=min(x[0] for x in parse_dump(popped)); end=max(x[0] for x in parse_dump(retired));
    assert isclose((end-start)/clock,q['ca_us'],abs_tol=.001)
    us=lambda tick:(tick-start)/clock
    seq=defaultdict(dict)
    for tick,pc,instr,op in parse_dump(popped):
        if pc-base in REDUCE:
            assert op=='RV_VCADD'
            seq[REDUCE[pc-base]].setdefault(instr,[]).append(tick)
    visits=[]
    for v in (0,64):
        for t in (0,1,2): visits.append(dict(v_start=v,token=t+1))
    for op,data in seq.items():
        assert len(data)==6*64 and all(len(rows)==2 for rows in data.values())
        ordered=sorted((min(records),iid) for iid,records in data.items())
        for i,visit in enumerate(visits):
            group=ordered[i*64:(i+1)*64]; visit[op]=dict(first_issue_us=us(group[0][0]),last_issue_us=us(group[-1][0]),
                dynamic_ids=[iid for _,iid in group], issue_ticks=[tick for tick,_ in group])
    xs=[x for x in json.loads(shard.read_bytes())['traceEvents'] if x['ph']=='X']; origin=min(x['ts'] for x in xs)
    assert isclose(origin,start/clock,abs_tol=.001)
    transfers=[]
    for x in sorted(xs,key=lambda a:a['ts']):
        if x['name']!='MOV_SRC_TO_DST_ALIGNv2': continue
        d=json.loads(x['args']['detail']);pipe=d['instr_type'];b=d['processed_bytes'];
        if pipe not in ('MTE2','MTE3'):continue
        item=dict(pipe=pipe,bytes=b,from_addr=hex(d['src_addr']),to_addr=hex(d['dst_addr']),burst_len=d['burst_len'],bursts=d['burst_number'],
                  src_stride=d['src_stride'],dst_stride=d['dst_stride'],offset=hex(int(x['args']['pc_addr'],16)-base),start_us=x['ts']-origin,end_us=x['ts']+x['dur']-origin)
        transfers.append(item)
    inputs=[d for d in transfers if d['pipe']=='MTE2' and d['bytes'] not in (32768,224)]; assert sorted(d['bytes'] for d in inputs)==[12,768,768,768,1536]
    metadata,=[d for d in transfers if d['pipe']=='MTE2' and d['bytes']==224]; reads=[d for d in transfers if d['pipe']=='MTE2' and d['bytes']==32768]
    assert len(reads)==2 and int(reads[1]['from_addr'],16)-int(reads[0]['from_addr'],16)==256
    state=[d for d in transfers if d['pipe']=='MTE3' and d['bytes']==32768];outs=[d for d in transfers if d['pipe']=='MTE3' and d['bytes']==128]
    assert len(state)==len(outs)==6 and all((x['bursts'],x['burst_len'])==(128,256) for x in reads+state)
    assert len(transfers)==20
    state_base=min(int(d['to_addr'],16) for d in state);out_base=min(int(d['to_addr'],16) for d in outs)
    for visit in visits:
        v=visit['v_start'];t=visit['token']-1
        by_tile=[d for d in state if int(d['to_addr'],16)==state_base+4*v]
        assert len(by_tile)==3
        visit['state_write']=by_tile[t]; visit['attn_write'],=[d for d in outs if int(d['to_addr'],16)==out_base+(t*128+v)*2]
        assert visit['decay_key']['last_issue_us'] < visit['update_query']['first_issue_us'] < visit['attn_write']['end_us']
    files=[shard,trace,popped,retired,dump/'pc_start_addr.txt',dump/'aicore_binary.o',qfile.parent/'run.log']
    source=next((ROOT/'toolchains/cann/9.2.1').rglob('recurrent_kda_arch35.h'));files.append(source)
    vendor=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/recurrent_kda_apt'/q['kernel'].rsplit('_',1)[0]; vendor=vendor.with_suffix('.o')
    device=elf_text(dump/'aicore_binary.o'); assert device==elf_text(vendor); files.append(vendor)
    sample=dict(elf_text_sha256=sha256(device).hexdigest(),kernel=q['kernel'],api=q['api'],shape=q['shape'],attrs=q['attrs'],chip=CHIP,tick_origin=start,tick_last=end,ticks_per_us=clock,
          load_address=hex(base),reduce_offsets={k:hex(pc) for pc,k in REDUCE.items()},predictions=preds, metadata=metadata,inputs=inputs,
          state_read=reads,visits=visits, artifacts={str(p.relative_to(ROOT)):digest(p) for p in files},
          scope='Each forecast covers a separate complete invocation. First observed pop per dynamic (PC, ID) is an issue landmark. Reduction-to-source mapping inferred from opcode signatures; issued landmarks do not represent entire vector service. DMA descriptor windows include dependencies.')
    (OUT/'state-steps.json').write_text(json.dumps(sample,indent=2)+'\n')
    return sample


def render(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    col={'control':'#a1acb0','input':'#258696','read':'#187887','decay_key':'#7151a1','update_query':'#4664ac','attn':'#da9662','state':'#ac482a'}
    fig,(a,b)=plt.subplots(2,1,sharex=True,figsize=(12,8.4),gridspec_kw={'height_ratios':[1,2.3]})
    fig.subplots_adjust(left=.21,right=.965,top=.83,bottom=.145,hspace=.50)
    fig.text(.21,.954,'950PR  RecurrentKda  |  3 tokens × 128 × 128 state',fontsize=21,weight='bold',color='#1b3140')
    fig.text(.21,.915,'1 AIV  ·  2 state slices of 64 V channels  ·  3 sequential updates per slice',fontsize=12,color='#425967')
    for ax in (a,b):
        ax.spines[:].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='x',color='#dbe5e8',lw=.75);ax.tick_params(axis='y',length=0,labelsize=10)
    a.set_title('Model  ·  complete calls at three sequence lengths',loc='left',fontsize=14,weight='bold');a.set_title(f"T=3   {s['predictions'][-1]['model_us']:.3f} µs",loc='right',fontsize=14,weight='bold')
    for i,p in enumerate(s['predictions']):
        y=2-i;a.barh(y,p['model_us'],height=.25,color=['#a3b9c1','#7193a0','#326879'][i]);a.plot(p['ca_us'],y,'|',ms=15,mew=2,color='#25333a');a.text(p['model_us']+.18,y,f"{p['model_us']:.2f}",va='center',fontsize=9,color='#2e4653')
    a.set_yticks([2,1,0],['T=1  ·  2 updates','T=2  ·  4 updates','T=3  ·  6 updates']);a.set_ylim(-.5,2.55)
    a.text(0,-.21,'Bars give whole-call forecasts. Dark ticks mark same-shape CA endpoints.',transform=a.transAxes,fontsize=9,color='#46606c')
    stage=s['predictions'][-1]['exposed_us']
    a.text(0,-.34,'T=3 exposed cost (µs)   Vector %.2f   ·   Scalar %.2f   ·   MTE2 %.2f   ·   MTE3 %.2f   ·   Other %.2f' % (stage['SIMD'],stage['SCALAR'],stage['MTE2'],stage['MTE3'],stage['other']),transform=a.transAxes,fontsize=9,color='#46606c')
    b.set_title('CA trace  ·  core0.veccore0   |   T=3',loc='left',fontsize=14,weight='bold');b.set_title(f"{s['predictions'][-1]['ca_us']:.3f} µs",loc='right',fontsize=14,weight='bold')
    def bar(y,d,c):
        b.broken_barh([(d['start_us'],d['end_us']-d['start_us'])],(y-.105,.21),facecolors=c,edgecolors='none')
    for i,d in enumerate([s['metadata'],*s['inputs']]):bar(7.7+i*.065,d,col['control' if i==0 else 'input'])
    for i,d in enumerate(s['state_read']):bar(7.13-i*.26,d,col['read']);b.text(d['end_us']+.13,7.13-i*.26,f"V{i*64}–{i*64+63}",fontsize=7.5,va='center',color='#186579')
    ticks=[8,7];labels=['Header + Q/K/V/g/β','State prefetch']
    for i,v in enumerate(s['visits']):
        y=5.68-i-(.28 if i>=3 else 0);ticks.append(y);labels.append(f"V{v['v_start']}–{v['v_start']+63}  ·  token {v['token']}")
        for dy,k in ((.20,'decay_key'),(.03,'update_query')):
            xs=[(z-s['tick_origin'])/s['ticks_per_us'] for z in v[k]['issue_ticks']];b.vlines(xs,y+dy-.07,y+dy+.07,lw=.55,color=col[k])
        bar(y-.17,v['attn_write'],col['attn']);bar(y-.36,v['state_write'],col['state'])
    b.axhline(3.02,color='#dae4e7',lw=.8);b.axvline(s['predictions'][-1]['ca_us'],color='#596e79',linestyle='--',lw=1)
    b.set_yticks(ticks,labels);b.set_ylim(-.25,8.55);b.set_xlim(0,24);b.set_xticks(range(0,25,2));b.set_xlabel('Device time from the first instruction (µs)',fontsize=11)
    handles=[Patch(color=col['input'],label='MTE2 inputs / state'),Line2D([],[],ls='',marker='|',ms=10,color=col['decay_key'],label='Decay + key reduction'),Line2D([],[],ls='',marker='|',ms=10,color=col['update_query'],label='Update + query reduction'),Patch(color=col['attn'],label='MTE3 output'),Patch(color=col['state'],label='MTE3 state')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.205,.901),ncol=5,frameon=False,fontsize=8.8,handlelength=1.35,columnspacing=1.35)
    fig.text(.21,.067,'Lower ticks show 64 dynamic reduce instructions of each kind per update. Bars show measured DMA descriptor windows.',fontsize=9.5,color='#425967')
    fig.text(.21,.037,'Each state write updates the same 32 KiB slice. The model estimates exposed resource time at whole-call resolution.',fontsize=9.5,color='#425967')
    fig.savefig(FIG,dpi=150,facecolor='white');plt.close(fig)

if __name__=='__main__':
    s=extract() if '--extract' in sys.argv else json.loads((OUT/'state-steps.json').read_text());render(s)
