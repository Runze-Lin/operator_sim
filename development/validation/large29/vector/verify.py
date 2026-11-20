"""Validate complete output and core sets. Partial exports never provide gold timing."""
from collections import Counter
import json
import re
from common import *
from development.validation.rms_ordered_verify.verify import checked,events,tables,text_bytes


def numerical(job,txt):
    op=job['op'];shape=job['shape'];r,c=shape[:2]
    lines=re.findall(r'^correctness [^\r\n]+',txt,re.M)
    if op=='attn_res_fwd':
        exact=[f'correctness shape={r}x{c} S={shape[2]} N={shape[3]} flags=cache,add,mix,nosave,nowrite groups=output,prefix,bank bad=0']
        checked(lines==exact,'complete prefill output, raw prefix, unchanged bank')
    elif op=='moe_gating_top_k':
        k=shape[-1];checked(f'T={r} E={c} K={k} dtype=FP32 norm_type=1 renorm=0 bias=true' in txt,'router contract')
        checked(lines==[f'correctness ids count={r*k} bad=0',f'correctness weights count={r*k} bad=0'],'router output and ids')
    else:
        expected=[r*c]
        if op=='quant':expected.append(r*((c+63)//64)*2)
        if op=='rmsnorm':expected.append(r)
        tag={'rmsnorm':f'case=rmsnorm api=aclnnRmsNorm R={r} C={c} dtype=BF16 epsilon=1e-6','cast_bf16_fp32':f'case=cast_bf16_fp32 api=aclnnCast R={r} D={c}','quant':f'case=dynamic_mx_quant rows={r} cols={c} input=BF16 output=E4M3FN block=32'}[op]
        checked(tag in txt and lines==[f'correctness count={v} bad=0' for v in expected],'shape and every numeric output')
    return lines


def qualify(job,d,cost):
    checked(cost['exit_code']==0 and not cost['stop_reason'],'completion or run budget')
    chip=job['chip'];op=job['op'];txt=(d/'run.log').read_text(errors='replace')
    checked(re.findall(r'^runtime_soc=(\S+)',txt,re.M)==[chip] and ('lpddr5_6400M_bailu.ini' if chip==PR else 'hbm3_8000M.ini') in txt,'physical memory and exact runtime SKU')
    checked('SMOKE_PASS' in txt and not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:deadlock|invalid instruction|execute_set_flag|execute_wait_flag)',txt,re.I),'runtime or instruction fault')
    correct=numerical(job,txt)
    if job['protocol']=='wrapper':
        starts={};fin={}
        for kind,dest in [('start',starts),('end',fin)]:
            pairs=re.findall(r'\[block_'+kind+r'\]\s+: AIV, task_id=0, core_id=(\d+), block_id=(\d+)',txt)
            checked(len(pairs)==len(set(v for v,_ in pairs))==len(set(b for _,b in pairs)),kind+' AIV roster')
            dest.update((int(c),int(b)) for c,b in pairs)
        checked('[TASK_DONE]' in txt and starts==fin and bool(fin) and set(fin.values())==set(range(len(fin))),'complete wrapper launch set')
        wins={}
        for f in (d/'log_ca').glob('core*_summary_log'):
            for c,l,a,b,block,dim in re.findall(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: 0, blkID: (\d+), blkDim: (\d+),.*? done',f.read_text()):
                c,l,a,b,block,dim=map(int,(c,l,a,b,block,dim));i=2*c+l-1
                checked(i not in wins and fin.get(i)==block and dim==len(fin) and b>a and l in (1,2),'wrapper identity/timestamps')
                wins[i]=(a,b)
        checked(set(wins)==set(starts),'all physical wrapper start/done')
        checked(len(set(re.findall(r'\[block_start\]\s+: AIV, task_id=(\d+)',txt)))==1,'single vector launch')
        rep={ 'qualified':True,'wrapper_us':(max(b for a,b in wins.values())-min(a for a,b in wins.values()))/1650,
            'instruction_us':None,'active_aiv':len(wins),'core_ticks':wins,'correctness':correct,'run_log_sha256':digest(d/'run.log')}
        if job['shape']==[32,4096]:
            archived=read(ROOT/'results/v0.29/replay.json')['rows'];r=next(r for r in archived if r['op']==op and r['shape']==job['shape'] and r['chip']==chip)
            # Recover the core count independently from this exact archived trace identity.
            checks=read(ROOT/'results/v0.19/workloads/cast-qualified.json')
            prev=next(x for x in checks if x['shape']==job['shape'] and x.get('chip',x.get('sku'))==chip and x['kernel']['trace_sha256'] in r['trace_sha256'])
            checked(len(wins)==prev['kernel']['active_aiv'],'same physical launch count as official control')
            rep.update(control_instruction_us=r['ca_us'],control_reference=prev['kernel']['trace'])
        return rep
    names=re.findall(r'Start profiling on kernel: (\S+)',txt)
    checked(len(names)==1 and 'All task success.' in txt and 'Profiling data parse finished.' in txt,'one parsed official kernel')
    name=names[0]
    if op=='attn_res_fwd':
        checked(name=='AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40021','actual cached prefill key')
        vendor,=(VENDOR/'op_impl/ai_core/tbe/kernel/ascend950/attn_res_fwd_apt').glob('*.o')
    elif op=='moe_gating_top_k':
        prefix=re.fullmatch(r'(MoeGatingTopK_[a-f0-9]+)_\d+',name);checked(prefix,'router kernel')
        vendor=OPS/'ops_transformer/moe_gating_top_k_apt'/(prefix[1]+'.o')
    else:
        checked(name.startswith('RmsNorm_'),'actual RMS kernel')
        candidates=list((OPS/'ops_nn/rms_norm').glob('*.o'));vendor=None
    tr,=d.glob('profile/OPPROF_*/simulator/trace.json');binary=tr.parents[1]/'dump/aicore_binary.o';actual=text_bytes(binary)
    if vendor is None:vendor=next((x for x in candidates if text_bytes(x)==actual),None)
    checked(vendor is not None and text_bytes(vendor)==actual,'exact installed kernel text')
    tab,=tables(txt);wins={};ends=Counter();pipes=Counter();names_per_core={}
    for e in events(tr):
        c=e['pid'];n=e['name'];a=round(1650*e['ts']);b=round(1650*(e['ts']+e['dur']))
        checked(b>=a and re.fullmatch(r'core\d+\.veccore\d+',c),'valid physical AIV')
        x=wins.setdefault(c,[a,b]);x[0]=min(x[0],a);x[1]=max(x[1],b)
        ends[c]+=n=='END';pipes[e['tid']]+=1
        names_per_core.setdefault(c,Counter())[n]+=1
    checked(set(wins)==set(tab) and wins,'official full-roster table')
    for c,(a,b) in wins.items():checked(ends[c]==1 and abs((b-a)/1650-float(tab[c][0]))<.025,'physical END and vendor duration '+c)
    return dict(qualified=True,instruction_us=(max(v[1] for v in wins.values())-min(v[0] for v in wins.values()))/1650,
        wrapper_us=None,active_aiv=len(wins),core_ticks=wins,correctness=correct,trace=rel(tr),trace_sha256=digest(tr),
        kernel=name,installed_object=rel(vendor),released_text_sha256=sha256(actual).hexdigest(),queues=pipes,
        vf_by_core={k:sum(count for n,count in v.items() if n.startswith('VFCALL')) for k,v in names_per_core.items()})
