"""Freeze a one-vs-two local batch boundary; never rerun the older large router."""
import argparse,fcntl,os,shutil,sys,time
from math import ceil
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import json
from evidence import ROOT,OUT,DT,PR,FUT,VENDOR,sha,read,rel,selected,merge_ops
from candidate import predict
sys.path.append(str(ROOT));from operator_model import estimate,load_chip
OLD=ROOT/'results/v0.29/large/vector';ORIG=ROOT/'development/validation/large29/vector';SLOT=ROOT/'results/v0.26/sink/raw/lock';RAW=ROOT/'runs/gating35';REG=OUT/'freeze.json';TARGET=[1536,896,16]

def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
    tmp=p.with_name('.'+p.name+'.'+str(os.getpid()));tmp.write_text(json.dumps(d,indent=2,allow_nan=False,ensure_ascii=False)+'\n');tmp.replace(p)

def original():
    sys.path.insert(0,str(ORIG));from acquire import capture
    from verify import qualify
    sys.path.remove(str(ORIG))
    return capture,qualify

def forecast(shape,chip):
    t=time.perf_counter_ns();r=predict(shape,load_chip(chip));r['candidate_query_ms']=(time.perf_counter_ns()-t)/1e6;r['public_us']=estimate('moe_gating_top_k',shape,chip=chip).latency_us;return r

def register():
    assert not REG.exists() and not RAW.exists();j=read(OLD/'freeze.json');olds=[x for x in j['jobs'] if x['op']=='moe_gating_top_k'];assert {x['chip'] for x in olds}=={DT,PR} and all(x['shape']==[4096,896,16] for x in olds)
    assert all(not (OLD/(x['case']+'.json')).exists() and not (ROOT/'runs/large29/vector'/x['case']).exists() for x in olds)
    cfg=read(ROOT/'configs/models/Kimi-K3.json')['text_config'];assert (cfg['num_experts'],cfg['num_experts_per_token'])==(896,16)
    assert not any(r['shape']==TARGET for r in read(OUT/'physical.json')['rows']);count=0
    for folder,dirs,fs in os.walk(ROOT/'runs'):
        dirs[:]=[d for d in dirs if d not in ('profile','simulator','log','log_ca','build','sources','dump','record','ca_workspace') and not d.startswith(('OPPROF_','PROF_'))]
        if 'run.log' not in fs:continue
        count+=1
        with (Path(folder)/'run.log').open('rb') as f:
            if any(b'T=1536 E=896 K=16 dtype=FP32 norm_type=1 renorm=0 bias=true' in line for line in f):raise ValueError('previous native acquisition')
    shapes=[TARGET,[4096,896,16],*[ [8*s,896,16] for s in (32768,131072,262144)]];fc=[dict(shape=s,forecasts={c:forecast(s,c) for c in (DT,PR,FUT)}) for s in shapes];reg=dict(utc=now(),cases=fc,job=dict(olds[0],case='kimi-1536',shape=TARGET,args=list(map(str,TARGET)),wall_s=14400,raw_gib=16),
        source=rel(OLD/'freeze.json'),existing={x['chip']:rel(OLD/(x['case']+'.json')) for x in olds},previous_sha=sha(OLD/'freeze.json'),
        slot=rel(SLOT),history_raw_logs_checked=count,scenario='Kimi 896 experts sigmoid, top16, FP32 scores and same FP32 bias for all tokens, one group. Independent B8×192 on one rank within 32Ki prefill. Strictly isolate the real aclnnMoeGatingTopK operation; router GEMM, dispatch and EP omitted.',
        budget='One DT and one PR on the pre-existing shared standalone physical lane, acquired and released once per endpoint. No overlap with Addmm/AttnRes/single KDA or change to 4096 owner. 4h or 16 GiB each, 48 GiB free headroom. --dump=false, official full time, all output golden.',
        hypothesis='DT/PR master rows 24 vs 28, E896 batch capacity 26. DT issues one logits DMA plus two outputs per critical AIV; PR issues two logits DMAs plus four outputs. Each row emits one VBS32, four VMS4V2 and four UB copies. Separate timing and mechanism gates.',
        acceptance='Full elapsed time requires native ABI + SKU + actual installed text + complete numerical outputs + unique physical END per vendor AIV; structural counters audited separately. Preserve failures and never replace prior measurements. No live information supplied to prices.',
        hashes={rel(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('candidate.py'),Path(__file__).with_name('evidence.py'),OUT/'prices.json',OUT/'physical.json',ROOT/olds[0]['runner'],ROOT/'development/workloads/moe_router_smoke.cpp',ORIG/'verify.py',ORIG/'acquire.py',VENDOR,ROOT/'configs/models/Kimi-K3.json']})
    with REG.open('x') as f:json.dump(reg,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def structural(path,shape,plan):
    t,e,k=shape;lanes=[];counter=0;tot=0
    for p in sorted(path.parent.glob('core*/trace.json')):
        v=selected(p);rd=[x for x in v if x[0]=='MOV_SRC_TO_DST_ALIGNv2' and x[3].get('src_mem')=='OUT' and x[3].get('burst_len')==4*e];wr=[x for x in v if x[0]=='MOV_SRC_TO_DST_ALIGNv2' and x[3].get('dst_mem')=='OUT'];sort=[x for x in v if x[0]=='VBS32'];r=len(sort);tot+=r;batches=[z[3]['burst_number'] for z in rd[1:]];end=[z for z in v if z[0]=='END'];assert len(end)==1
        merges=sum(x[0]=='VMS4V2' for x in v);copies=sum(x[0]=='MOV_UB_TO_UB' for x in v);pred_batches=[min(plan['batch_capacity'],r-j) for j in range(0,r,plan['batch_capacity'])];expected,c=merge_ops(e)
        ok=(len(rd)>1 and rd[0][3]['burst_number']==1 and batches==pred_batches and len(wr)==2*len(batches) and merges==r*len(expected) and copies==r*c)
        counter+=ok;lanes.append(dict(core=p.parent.name,rows=r,observed_x_batch_rows=batches,expected_x_batch_rows=pred_batches,mmad=0,vbs32=r,vms4v2=merges,ub_copy=copies,writes=len(wr),matches=ok))
    return dict(trace_sha256=sha(path),all_lanes_match=counter==len(lanes)==plan['active_aiv'] and tot==t,active=len(lanes),rows=tot,cores=lanes)

def run():
    f=read(REG);assert all(sha(ROOT/p)==v for p,v in f['hashes'].items());RAW.mkdir(exist_ok=True);run_ca,qual=original();os.nice(15)
    def state(phase,**kw):write(OUT/'state.json',dict(utc=now(),pid=os.getpid(),phase=phase,freeze_sha256=sha(REG),**kw))
    with (RAW/'owner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            p=OUT/(tag+'.json');d=RAW/tag;j=dict(f['job'],chip=sku)
            if p.exists():continue
            if not (d/'cost.json').exists():
                state('waiting for existing physical CA slot',chip=sku,slot=rel(SLOT))
                with SLOT.open('r+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    if d.exists():state('preserving interrupted run',chip=sku);return
                    if shutil.disk_usage(ROOT).free<48*2**30:state('waiting for disk headroom',chip=sku);return
                    state('acquiring',chip=sku);run_ca(j,d)
            cost=read(d/'cost.json');state('verifying after slot release',chip=sku)
            try:
                if cost['exit_code'] or cost['stop_reason'] or cost['started_utc']<=f['utc']:raise ValueError('capture failed/budget/chronology')
                v=qual(j,d,cost);assert v['qualified']
            except (ValueError,AssertionError,StopIteration,TypeError,KeyError,OSError) as exc:v=dict(qualified=False,reason=repr(exc))
            aud=None
            if v['qualified']:
                try:aud=structural(ROOT/v['trace'],TARGET,f['cases'][0]['forecasts'][sku])
                except (ValueError,AssertionError,KeyError,OSError) as exc:aud={'all_lanes_match':False,'error':repr(exc)}
            write(p,dict(shape=TARGET,chip=sku,cost=cost,verdict=v,mechanism=aud,freeze_sha256=sha(REG)))
            from report import refresh
            refresh()
        state('terminal')

if __name__=='__main__':{'freeze':register,'run':run}[sys.argv[1]]()
