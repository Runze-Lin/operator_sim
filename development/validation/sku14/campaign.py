"""Preregister PR shapes by DT record ID before predicting or opening any PR labels."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.14/sku'
RUNS = ROOT/'runs/sku14'
PR = 'Ascend950PR_957b'
DT = 'Ascend950DT_9582'

def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load(path): return json.loads(path.read_text())
def utc(): return datetime.now(timezone.utc).isoformat()
def write(path, obj, exclusive=False):
    with path.open('x' if exclusive else 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

GROUPS = {
'quant': [('q2_cal_tiny','single-row tiny'),('q_cal_small','small row tile'),('q_test_tail','non-power-of-two width'),('q2_test_wide','single wide row'),('q_test_wave','uneven active count'),('q2_cal_saturation','65 rows; check grouped-row dispatch'),('q4_test_flash_decode','Q=9 wide verify'),('q4_test_pro_decode','wide batch verify'),('q3_test_one_ub','large; DT one UB regime'),('q2_test_ub_loop','large; beyond DT one UB regime'),('q_cal_decode','17-token verify'),('q3_test_two_ub','large multiple UB tiles')],
'matmul': [('m_cal_decode','single decode token'),('m_cal_small','small aligned'),('m_test_tail','M tail'),('m_test_prefill','intermediate output tiles'),('m2_cal_k_tail','K=129 tail'),('m2_cal_skinny_k','M=1 K=1024'),('m2_test_k_work','M=17 K=1024'),('m2_cal_full','aligned K and N=512'),('matmul_256_512_256','double M, 16+ candidate cores'),('matmul_512_512_256','large; 32 DT AICs'),('matmul_31_64_257','K=257 and M tail'),('matmul_128_192_256','N tail')],
'fa': [('f_cal_decode','single-token decode'),('f_test_verify','17-token verify'),('f2_cal_d128','head width=128'),('f2_cal_heads4','multi-head'),('f2_cal_q128','before Q tile boundary'),('f2_test_q129','after Q tile boundary'),('f2_cal_longkv','KV=1025 tail'),('f3_test_longkv','KV=1153 longer loop'),('f4_test_batch4','batch=4 Q=9 Hdim=128'),('fa_9_4096_64_1_1','large; 4096-context Q=9 loop'),('f3_test_d128_tail','head width 128 KV=257'),('f2_test_heads8','8 heads')]}

CONTRACTS = {
    'quant':{'api':'aclnnDynamicMxQuant','family':'ops-nn','axes':['rows','cols'],'dtype':'bf16','attrs':{'dst':'e4m3fn','scale_dtype':'e8m0','axis':-1,'block':32,'rounding':'rint'},'golden':'all output and scale bytes'},
    'matmul':{'api':'aclnnMatmul','family':'ops-nn','axes':['M','N','K'],'dtype':'bf16','attrs':{'bias':False,'transpose':False,'layout':'ND','cube_math_type':0},'golden':'every BF16 output vs FP64; atol=0.01, rtol=0.005'},
    'fa':{'api':'aclnnFlashAttentionScoreV2','family':'ops-transformer','axes':['Q','KV','D','B','H'],'dtype':'bf16','attrs':{'layout':'SBH','causal':False,'keep_prob':1.0},'golden':'every BF16 attention output vs FP64; atol=0.002, rtol=0.01; does not validate auxiliary statistics'},
}

def build():
    runners = {sha(ROOT/p):str(p) for p in ['runs/model-v1/three_ops-frozen','runs/model-v4/three_ops-v03','build/smoke/three_ops']}
    old_records={}
    for f in sorted((ROOT/'runs').glob('model-v[1234]/*/result.json')):
        d=load(f)
        if d['status']=='ok': old_records[d['case']['id']]=(f,d)
    cases=[]
    for op,group in GROUPS.items():
        for i,(key,coverage) in enumerate(group):
            if key in old_records:
                f,d=old_records[key]; args=d['case']['shape']; binary=d['identity']['host_binary_sha256']; backend='msopprof'; dt_s=d['wall_seconds']; raw=d['raw_bytes']
            else:
                f=ROOT/(('results/v0.8/core/'+key+'.json') if op=='fa' else ('runs/v8cube/'+key+'/result.json'))
                d=load(f); assert d['status']=='ok'; args=d['shape']; binary=sha(ROOT/'build/smoke/three_ops'); backend='npusim'; dt_s=d.get('wall_s',d.get('wall_seconds'));raw=d.get('raw_bytes',d.get('measurement',{}).get('db_bytes'))
            shape=args+([1,1] if op=='fa' and len(args)==3 else [])
            cases.append(dict(id=key,op=op,shape=shape,cli_shape=args,coverage=coverage,role='independent_pr',priority=('primary' if i<10 else 'reserve'),dt_result=str(f.relative_to(ROOT)),dt_backend=backend,runner=runners[binary],host_binary_sha256=binary,dt_wall_seconds=dt_s,dt_raw_bytes=raw,wall_limit_s=(300 if dt_s>105 else 200),raw_limit_mib=(1900 if op=='quant' and max(shape)>=4096 else 950)))
    prior=[]
    for name in ('pairs','holdout'):
        prior.extend(load(ROOT/f'results/v0.13/sku/{name}.json'))
    seen={(r['op'],tuple(r['shape']+([1,1] if r['op']=='fa' and len(r['shape'])==3 else []))) for r in prior}
    for r in cases:
        key=(r['op'],tuple(r['shape'])); assert key not in seen; seen.add(key)
    return dict(utc=utc(),sku=PR,role='independent only on PR; paired historical DT shapes',cases=cases,contracts=CONTRACTS,max_primary=30,max_reserves=6,max_workers=4,max_total_seconds=2400,max_total_raw_mib=12500,retry_policy='Only preregistered reserve in within-op order after failed qualification/infrastructure; no substitution based on prediction error. Frozen shape budget shared across all attempts. No DT CA reruns.')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True); write(OUT/'budget.json',build(),True)
