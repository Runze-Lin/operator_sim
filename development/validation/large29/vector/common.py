"""Frozen scenario definitions and I/O for paired large Vector measurements."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from operator_model.attn_res_prefill_plan import prefill_service_plan
from build_host import HOST, VENDOR
OUT=ROOT/'results/v0.29/large/vector'
RAW=ROOT/'runs/large29/vector'
FREEZE=OUT/'freeze.json'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
PREFILL=dict(addend=True,write_idx=-1,materialize=False,optimize_prefill=True,mix=True,output_norm=True)
OPS=ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950'

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_bytes())
def digest(p):
    with p.open('rb') as f:
        h=sha256()
        while block:=f.read(1048576):h.update(block)
    return h.hexdigest()
def save(path,obj,*,new=False):
    if new:
        with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def rel(p):return str(p.relative_to(ROOT))
def forecast(op,shape,chip,attrs=None):
    p=estimate(op,shape,attrs,chip=chip)
    resource=estimate(op,shape,attrs,chip=chip,timing_mode='resource') if not p.projection else p
    plan=dict(resource.projection['target_plan'])
    plan.pop('ub_tile_rows',None)
    return dict(public_us=p.latency_us,physical_plan=plan,active_aiv=p.active_aiv,
                extrapolation=chip==FUT,
                ideal_service=prefill_service_plan(shape,chip) if op=='attn_res_fwd' and attrs==PREFILL else None)

def freeze():
    if FREEZE.exists() or RAW.exists():raise FileExistsError(FREEZE)
    cap=3600;vec=ROOT/'build/smoke/large_vector';cast=ROOT/'build/smoke/large_cast20';router=ROOT/'build/smoke/moe_router19'
    jobs=[]
    def enroll(op,shape,attrs,entry,argv,skus,hours,gib,protocol,scenario,gate=None):
        prices={s:forecast(op,shape,s,attrs) for s in (DT,PR,FUT)}
        for s in skus:
            mono='-'.join(map(str,shape))
            jobs.append(dict(case=f'{protocol}-{op}-{mono}-{s}',op=op,chip=s,shape=shape,attrs=attrs,prices=prices,
                runner=rel(entry),args=list(map(str,argv)),wall_s=hours*cap,raw_gib=gib,protocol=protocol,
                scenario=scenario,control_required=gate))
    enroll('moe_gating_top_k',[4096,896,16],{},router,[4096,896,16],[DT,PR],5,64,'instruction','Kimi B8 × 512 scheduled tokens, 896 experts top16, one sigmoid bias group')
    enroll('attn_res_fwd',[512,7168,8,1],PREFILL,HOST,[512,1],[DT,PR],4,64,'instruction','Kimi prefill chunk of 512 scheduled tokens, layers 13–23, one valid cached residual, source cache flags')
    enroll('rmsnorm',[8192,5120],{},vec,['rmsnorm',8192,5120],[DT,PR],4,64,'instruction','Flash B4 × 2048 scheduled tokens, hidden 5120')
    # Existing long jobs are linked by the report, never acquired here.
    enroll('cast_bf16_fp32',[32,4096],{},cast,[32,4096],[PR],1,4,'wrapper','PR native dual-SKU wiring control checked against an archived instruction window')
    ctrl='wrapper-cast_bf16_fp32-32-4096-'+PR
    enroll('cast_bf16_fp32',[262144,5120],{},cast,[262144,5120],[PR],36,72,'wrapper','Flash full B8 × 32 Ki router-input cast',ctrl)
    enroll('quant',[262144,7168],{},vec,['quant',262144,7168],[PR],36,72,'wrapper','Kimi full B8 × 32 Ki mixed QKV MXFP8 preprocessing',ctrl)
    enroll('rmsnorm',[262144,5120],{},vec,['rmsnorm',262144,5120],[DT,PR],36,72,'wrapper','Flash full B8 × 32 Ki RMSNorm',None)
    # The repaired full Flash PR RMS native launch also depends on the PR control.
    jobs[-1]['control_required']=ctrl
    kernels=list((OPS/'ops_transformer/moe_gating_top_k_apt').glob('*.o'))+list((OPS/'ops_nn/rms_norm').glob('RmsNorm_normal_all_bf16_high_performance.o'))
    inputs=[vec,cast,router,HOST,HOST.with_suffix('.cpp'),ROOT/'development/workloads/moe_router_smoke.cpp',
            ROOT/'development/workloads/attention_residual_smoke.cpp',ROOT/'smoke/large_vector.cpp',ROOT/'development/validation/large20/cast.cpp',
            ROOT/'operator_model/attn_res_prefill_plan.py',ROOT/'operator_model/attention_residual.py',
            ROOT/'operator_model/attention_residual_timing.py',ROOT/'operator_model/moe_router.py',*kernels,
            *[VENDOR/p for p in read(ROOT/'results/v0.19/workloads/attn-res-build.json')['vendor']],
            *[ROOT/'configs/chips'/(s+'.json') for s in (DT,PR,FUT)],
            *[Path(__file__).with_name(n) for n in ('build_host.py','native-entry.sh','acquire.py','common.py','verify.py')],
            ROOT/'build/capture/libpr15.so',ROOT/'build/capture/libsim_soc_adapter.so']
    save(FREEZE,dict(utc=now(),jobs=jobs,inputs_sha256={rel(p):digest(p) for p in inputs},
        evidence='One terminal attempt per case. Full instruction: release .text, exact SKU, all numerical outputs, trace vs vendor active core roster and per-core END. Wrapper: full hardware start/done and host numerical, no instruction attribution. PR control [32,4096] qualifies wiring via runtime, actual LPDDR controller, archived official launch core count, matching full output and complete native barrier/window. The two time intervals have different endpoints; both are reported without an equality tolerance. Failed control leaves dependent PR native unattempted. No kernel name proof through native wrapper alone.',
        queue='One endpoint per acquisition of runs/large29/shared.lock; never hold it when verifying exported trace.'),new=True)
