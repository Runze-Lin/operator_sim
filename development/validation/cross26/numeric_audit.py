"""CPU-only numeric diagnosis. The stored BF16 arrays and acceptance gates are read-only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_triangular

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/kda_v23'))
from reference import bf16,prepare

OUT=ROOT/'results/v0.26/kda'
RAW=ROOT/'results/v0.24/kda-prefetch/raw'
SRC=next((ROOT/'results/chunk-recovery/fla2691/raw/fla_npu').rglob('chunk_kda_fwd_prepare/arch35/chunk_kda_fwd_prepare_vec.h'))
ORACLE=ROOT/'results/kda-v23/raw/vendor-prepare-oracle.py'
SKUS=('Ascend950DT_9582','Ascend950PR_957b')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def reference(t,precision,h=0):
    # Public ATK _reference uses FP64 gate, cumsum, scores and a two-block solve.
    # The wheel's StageV0Vf / ExpPair / StageV3Vf use the FP32 order below.
    hw=precision in ('device_fp32','device_gate_only','device_f32_dot')
    ty=np.float32 if hw else np.float64
    freq=np.float32 if precision in ('device_fp32','device_f32_dot') else np.float64
    idx=np.arange(t)[:,None]; chan=np.arange(128);i=idx*128+chan
    q=bf16(((i*7+3)%29-14)/64).astype(ty)
    k=bf16(((i*11+5)%31-15)/128).astype(ty)
    raw=bf16(-5+((idx+3*chan+h)%11-5)/8).astype(ty)
    plus=raw+(((chan+h)%7-3)/32).astype(ty)
    head_scale=np.exp(ty(h*.125))
    beta=((3+idx[:,0]%5)/16).astype(freq)
    if precision=='fp32_gate_step_only':logit=(np.float32(-5)*(np.float32(1)/(np.float32(1)+np.exp(-np.exp(np.float32(h*.125))*plus.astype(np.float32))))).astype(ty)
    elif hw:logit=ty(-5)*(ty(1)/(ty(1)+np.exp(-head_scale*plus)))
    else:logit=ty(-5)/(ty(1)+np.exp(-head_scale*plus))
    aq=np.zeros((t,64));ak=np.zeros_like(aq);operand={}
    for start in range(0,t,64):
        n=min(64,t-start)
        gv=logit[start:start+n]
        if precision=='fp32_scaled_carry_only':g=np.cumsum(gv.astype(np.float32)*np.float32(1/np.log(2)),axis=0,dtype=np.float32).astype(ty)
        elif hw:g=np.cumsum(gv*ty(1/np.log(2)),axis=0,dtype=ty)
        else:g=np.cumsum(gv,axis=0,dtype=ty)*ty(1/np.log(2))
        tri=np.zeros((64,64),dtype=freq)
        for a in range(0,n,16):
            b=min(a+16,n)
            ref=g[(a+b)//2]
            def factor(z):
                clipped=np.clip(z,-126,120)
                return np.exp(clipped*ty(np.log(2))) if hw else np.exp2(clipped)
            ql=bf16(q[start+a:start+b]*factor(g[a:b]-ref)).astype(freq)
            kl=bf16(k[start+a:start+b]*factor(g[a:b]-ref)).astype(freq)
            right_uncast=k[start:start+b]*factor(ref-g[:b]);kr=bf16(right_uncast).astype(freq)
            qdot=ql@kr.T;kdot=kl@kr.T
            projected=qdot*freq(1/np.sqrt(128))
            aq[start+a:start+b,:b]=bf16(projected)
            tri[a:b,:b]=kdot*beta[start+a:start+b,None]
            if start==192 and a==48:
                for row,col,side in [(51,44,'Q'),(50,40,'K')]:
                    if row>=b:continue
                    left=(ql if side=='Q' else kl)[row-a]
                    operand[side]=dict(right_uncast=right_uncast[col].copy(),ref_global=start+(a+b)//2,left=left.copy(),right=kr[col].copy(),dot=float((left*kr[col]).sum(dtype=np.float64)),gate_values=g[row].copy())
        aq[start:start+n,:n]=np.tril(aq[start:start+n,:n])
        tri=np.tril(tri,-1);eye=np.eye(32,dtype=freq)
        if hw and precision=='device_fp32':
            invs=[]
            for axis in (0,32):
                x=eye.copy();s=tri[axis:axis+32,axis:axis+32]
                for row in range(1,32):
                    for src in range(row):
                        x[row]=x[row]-s[row,src]*x[src]
                invs.append(x)
            x0,x1=invs
        else:
            x0=solve_triangular(eye+tri[:32,:32],eye,lower=True)
            x1=solve_triangular(eye+tri[32:,32:],eye,lower=True)
        xx=np.zeros_like(tri);xx[:32,:32]=x0;xx[32:,32:]=x1
        xx[32:,:32]=-x1@(tri[32:,:32]@x0)
        ak[start:start+n,:]=bf16(xx[:n])
        if start==192:
            operand['lower']=tri.copy()
    return (aq,ak),operand


def compare(actual,pred,point):
    dy=np.abs(actual-pred);den=2e-6+np.abs(pred)/64
    js=np.argwhere(dy>den)
    return dict(count=actual.size,bitwise_equal=int((actual==pred).sum()),out_of_frozen_tolerance=int(len(js)),
                raw_at_point=float(actual[point]),prediction_at_point=float(pred[point]),
                misses=[list(map(int,j)) for j in js[:10]])


def analyze(t,row,col,label):
    slug='supplement' if t==257 else 'prospective26'
    arr=[]
    for sku in SKUS:
        folder=RAW/f'{slug}-{sku}-{t}-1'
        arr.append({s:(np.fromfile(folder/(s+'.bf16'),dtype='<u2').astype('<u4')<<16).view('<f4').reshape(t,64) for s in ('aqk','akk')})
    a=arr[0]
    pub,math=prepare(t,1)
    result=dict(archived_files={sku:{s:{'path':str((RAW/f'{slug}-{sku}-{t}-1'/(s+'.bf16')).relative_to(ROOT)),'sha256':sha(RAW/f'{slug}-{sku}-{t}-1'/(s+'.bf16'))}for s in a} for sku in SKUS},
        coordinate=[0,row,col],tensor=label,dt_pr_arrays_byte_identical={s:bool(np.array_equal(a[s],arr[1][s]))for s in a},
        reference_variants={},frozen_value=float(pub[0 if label=='aqk' else 1][0,row,col]),
        fp64_mathematical_value=float(math[0 if label=='aqk' else 1][0,row,col]))
    ops={}
    for method in ('vendor_atk_fp64','fp32_gate_step_only','fp32_scaled_carry_only','device_gate_only','device_f32_dot','device_fp32'):
        (aq,ak),ops[method]=reference(t,method)
        result['reference_variants'][method]={s:compare(a[s],pr,(row,col))for s,pr in zip(('aqk','akk'),(aq,ak))}
        if method=='vendor_atk_fp64':
            result['published_formula_vs_local_frozen']={s:dict(mismatches=int(np.count_nonzero(pr!=target[0])),max_abs=float(np.max(abs(pr-target[0]))))for s,pr,target in zip(('aqk','akk'),(aq,ak),pub)}
    side='Q' if label=='aqk' else 'K'
    old=ops['vendor_atk_fp64'][side];new=ops['device_fp32'][side]
    ch=np.nonzero((old['left']!=new['left']) | (old['right']!=new['right']))[0]
    result['dot_trace']=dict(reference_row=old['ref_global'],changed_bf16_channel=[int(z)for z in ch],
            old_dot=old['dot'],new_dot=new['dot'],sum_absolute_terms=float(np.abs(new['left']*new['right']).sum(dtype=np.float64)),
            changed_products=[dict(channel=int(z),old_left=float(old['left'][z]),new_left=float(new['left'][z]),
                    old_right=float(old['right'][z]),new_right=float(new['right'][z]),old_right_before_cast=float(old['right_uncast'][z]),new_right_before_cast=float(new['right_uncast'][z]),midpoint=float((old['right'][z]+new['right'][z])/2)) for z in ch])
    if label=='akk':
        p,q=ops['vendor_atk_fp64']['lower'],ops['device_gate_only']['lower']
        u,v=row-192,col
        def inspect(values):
            h=solve_triangular(np.eye(32)+values[32:,32:],np.eye(32),lower=True)
            z=h[u-32,v-32]
            return dict(fp64=float(z),bf16=float(bf16([z])[0]))
        swap=p.copy();swap[u,v]=q[u,v]
        revert=q.copy();revert[u,v]=p[u,v]
        result['counterfactual_same_fp64_solve']=dict(published=inspect(p),only_one_changed_lkk_entry=inspect(swap),
                 all_gate_changes=inspect(q),restore_only_that_entry=inspect(revert),
                 nonidentical_entries_in_dependency_triangle=[dict(row=i,col=j,delta=float(q[i,j]-p[i,j]))
                    for i in range(v,u+1)for j in range(v,i)if abs(q[i,j]-p[i,j])>1e-12])
        result['diagonal32_inverse_inputs']=dict(local_row=row-192-32,local_col=col-32,
          largest_raw_lkk_change=float(np.max(abs(p[32:51,32:51]-q[32:51,32:51]))),
          lkk_target_old=float(p[row-192,col]),lkk_target_new=float(q[row-192,col]))
    return result


def audit():
    out=dict(study='Read-only CPU audit; measured CA, golden and tolerances unchanged. No PyTorch/ATK runtime is installed. ATK formula reproduced via NumPy/SciPy, not a rerun of the upstream executor.',
             fingerprints={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ORACLE,SRC,SRC.with_name('chunk_kda_fwd_prepare_cube.h'),SRC.parent.parent/'chunk_kda_fwd_prepare_policy.h',ROOT/'development/validation/kda_v23/reference.py']},
             cases={'257':analyze(257,243,44,'aqk'),'248':analyze(248,242,40,'akk')})
    p=OUT/'numeric-audit.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    for t,r in out['cases'].items():
        print(t,r['dt_pr_arrays_byte_identical'],r['published_formula_vs_local_frozen'],r['dot_trace'])
        for v,q in r['reference_variants'].items():print(v,[(s,a['prediction_at_point'],a['bitwise_equal'],a['out_of_frozen_tolerance'])for s,a in q.items()])

def matrices(t,h,variant):
    groups=[reference(t,variant,i)[0] for i in range(h)]
    return {s:np.stack([x[i] for x in groups]) for i,s in enumerate(('aqk','akk'))}


def serialized(a,fmt):
    if fmt=='bf16':return (a.astype('<f4').view('<u4')>>16).astype('<u2').tobytes()
    return a.astype('<f8').tobytes()


def diff_full(actual,pred):
    delta=np.abs(actual-pred);limit=2e-6+np.abs(pred)/64
    i=tuple(map(int,np.unravel_index(np.argmax(delta),delta.shape)))
    return dict(count=int(delta.size),different_bits=int((actual!=pred).sum()),bad=int((delta>limit).sum()),
        max_abs=float(delta[i]),max_abs_coordinate=list(i),actual_at_max_abs=float(actual[i]),reference_at_max_abs=float(pred[i]))


def counter_control(t,h,slug,reg):
    oracle={key:matrices(t,h,v) for key,v in [('published','vendor_atk_fp64'),('device','device_fp32')]}
    result={}
    for sku in SKUS:
        folder=RAW/f'{slug}-{sku}-{t}-{h}'
        original=ROOT/'results/v0.24/kda-prefetch/supplement'/(folder.name+'.json') if slug=='supplement' else OUT/(folder.name+'-official.json')
        record=json.loads(original.read_bytes())
        result[sku]=dict(original=str(original.relative_to(ROOT)),official_qualified=record['qualified'],expected_hashes=reg['oracles'][f'{t}-{h}'],groups={})
        for name in ('aqk','akk'):
            observed=folder/(name+'.bf16')
            if sha(observed)!=record['actual_aux_bf16'][name]['sha256']:raise ValueError('measured output fingerprint mismatch')
            actual=(np.fromfile(observed,dtype='<u2').astype('<u4')<<16).view('<f4').reshape(h,t,64)
            scored={}
            for key,values in oracle.items():
                data=serialized(values[name],'bf16' if key=='device' else 'f8')
                if hashlib.sha256(data).hexdigest()!=reg['oracles'][f'{t}-{h}'][name][key+'_sha256']:raise ValueError('forecast changed')
                scored[key]=diff_full(actual,values[name])
            result[sku]['groups'][name]={'measured_sha256':sha(observed),**scored}
    return result


def register():
    from datetime import datetime,timezone
    import kda_aligned as aligned
    target=OUT/'numeric-audit-freeze.json'
    if target.exists():raise FileExistsError(target)
    t,h=259,2;new=RAW/'golden'/f'{t}-{h}'
    if new.exists() or any(d.is_dir() and d.name.endswith(f'-{t}-{h}') for d in RAW.iterdir()):raise ValueError('prior acquisition')
    assigned={'193-2':(193,2),'259-2':(t,h)};oracles={}
    for label,(tt,hh) in assigned.items():
        published=matrices(tt,hh,'vendor_atk_fp64');device=matrices(tt,hh,'device_fp32');frozen,_=prepare(tt,hh)
        oracles[label]={}
        for name,expected in zip(('aqk','akk'),frozen):
            if not np.array_equal(expected,published[name]):raise ValueError('published CPU formula differs from frozen harness')
            oracles[label][name]=dict(published_sha256=hashlib.sha256(serialized(expected,'f8')).hexdigest(),
             device_sha256=hashlib.sha256(serialized(device[name],'bf16')).hexdigest(),reference_crosscheck=diff_full(device[name],expected))
    new.mkdir()
    pub=matrices(t,h,'vendor_atk_fp64')
    for name in ('aqk','akk'):(new/(name+'.bin')).write_bytes(serialized(pub[name],'f8'))
    parent=aligned.read(aligned.REG);shape=[1,t,h,h,128,128];fore={s:aligned.predict(shape,s) for s in (*SKUS,'Ascend970DT')}
    pins={p:c for p,c in parent['source_hashes'].items() if '/raw/golden/' not in p}
    for p in (Path(__file__),ORACLE,SRC,SRC.parent/'chunk_kda_fwd_prepare_cube.h',SRC.parent.parent/'chunk_kda_fwd_prepare_policy.h',
              new/'aqk.bin',new/'akk.bin',aligned.REG):pins[str(p.relative_to(ROOT))]=sha(p)
    for p,hs in pins.items():
        if sha(ROOT/p)!=hs:raise ValueError(p)
    clean={k:parent[k] for k in ('published_device_text','budget_per_capture_seconds','budget_per_capture_bytes','min_free_bytes','min_available_ram_bytes','max_load_per_online_cpu')}
    reg=dict(**clean,utc=datetime.now(timezone.utc).isoformat(),source_hashes=pins,oracles=oracles,
        scope='Frozen before reading H2 control actual buffers and before new 259/H2 CA. Host retains original FP64 public-algorithm golden; mixed-precision outputs scored separately offline and cannot qualify a failed stage.',
        cases=[dict(id=f'numeric26-{sku}-{t}-{h}',sku=sku,shape=shape,physical_memory=aligned.ca.MEMORY[sku],oracle=dict(attention=t*h*128,aqk=t*h*64,akk=t*h*64,state_vk=h*128*128),predictions=fore)for sku in SKUS])
    with target.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')


def control():
    reg=json.loads((OUT/'numeric-audit-freeze.json').read_bytes())
    if sha(Path(__file__))!=reg['source_hashes'][str(Path(__file__).relative_to(ROOT))]:raise ValueError('source changed')
    out=counter_control(193,2,'supplement',reg)
    (OUT/'numeric-audit-control.json').write_text(json.dumps(out,indent=2)+'\n')


def collect():
    import fcntl
    import time
    import kda_aligned as aligned
    rp=OUT/'numeric-audit-freeze.json';reg=aligned.read(rp);status=OUT/'numeric-audit-state.json'
    for p,h in reg['source_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError(p)
    while aligned.read(OUT/'aligned-state.json')['state']!='complete':
        aligned.ca.save(status,dict(state='waiting for predecessor T192/H4',utc=aligned.ca.when()));time.sleep(25)
    aligned.ca.BIN=aligned.HOST
    with (RAW/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for case in reg['cases']:
            receipt=OUT/(case['id']+'-official.json');folder=RAW/case['id']
            if receipt.exists():continue
            while not aligned.capacity(reg)[1]:
                aligned.ca.save(status,dict(state='waiting for CA host headroom',case=case['id'],utc=aligned.ca.when()));time.sleep(45)
            if not (folder/'cost.json').exists():
                if folder.exists():raise ValueError('incomplete earlier attempt preserved')
                folder.mkdir()
            aligned.ca.save(status,dict(state='acquisition or verification',case=case['id'],utc=aligned.ca.when()))
            cost=aligned.ca.capture(reg,case,folder,reg['budget_per_capture_seconds'])
            obs=aligned.qualify.verify(reg,case,folder,cost)
            files={k:folder/(k+'.bf16') for k in ('aqk','akk')}
            rec={k:dict(bytes=p.stat().st_size,sha256=sha(p)) for k,p in files.items() if p.exists()}
            obs['qualified']=obs['qualified'] and all(k in rec and rec[k]['bytes']==case['oracle'][k]*2 for k in files)
            aligned.ca.save(receipt,dict(**obs,case=case,cost=cost,registration_sha256=sha(rp),actual_aux_bf16=rec))
            if len(rec)!=2:aligned.ca.save(status,dict(state='output missing; preserved and stopped',case=case['id']));return
        score=counter_control(259,2,'numeric26',reg)
        aligned.ca.save(OUT/'numeric-audit-prospective.json',score)
        aligned.ca.save(status,dict(state='complete',utc=aligned.ca.when(),original_qualification=[x['official_qualified']for x in score.values()]))


if __name__=='__main__':
    {'audit':audit,'register':register,'control':control,'collect':collect}[sys.argv[1] if len(sys.argv)>1 else 'audit']()
