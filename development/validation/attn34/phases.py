"""Learn only common 40020 instruction-delimited subphases to transfer to the sibling branch."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json,math,statistics,sys
import numpy as np
from scipy.optimize import nnls
R=Path(__file__).resolve().parents[3];O=R/'results/v0.34/attn'
sys.path.insert(0,str(R/'development/validation/coverage33'))
from rms_archive import moves,endpoint
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
def read(p):return json.loads(p.read_bytes())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def inputs():
    for n in ('attn-res-micro-ca','attn-res-holdout','attn-res-extension-holdout'):
        p=R/'results/v0.19/workloads'/(n+'.json')
        for row in read(p):
            if row.get('qualified'):yield p,row,True
    for subdir in ('receipts','retries'):
        for p in sorted((R/'results/v0.21/fusion'/subdir).glob('attn_res_fwd*.json')):
            row=read(p)
            if row.get('qualified'):yield p,row,False
def phase_data(path,row,train):
    s=row['kernel'];c=max(s['windows'],key=lambda name:s['windows'][name][1]);sh=row['shape'];t,h,_,n=sh;chip=row['chip']
    tr=R/s['trace'];p=tr.parent/c/'trace.json';m=endpoint(p)
    offset=max(z['retire'] for z in m.values())-round(1650*s['windows'][c][1])
    items=[]
    for e in moves(p):
        d=json.loads(e['args']['detail'])
        if d['instr_type'] not in ('MTE2','MTE3') or d['processed_bytes']!=2*h or any(d[k] for k in ('is_sync','is_wait','is_duration_excluded')):continue
        items.append(dict(queue=d['instr_type'],start=round(1650*e['ts'])-offset,retire=round(1650*(e['ts']+e['dur']))-offset))
    reads=sorted((i for i in items if i['queue']=='MTE2'),key=lambda i:i['start'])
    writes=sorted((i for i in items if i['queue']=='MTE3'),key=lambda i:i['start'])
    assert len(reads)==n+5 and len(writes)==4,(sh,reads,writes)
    w0,p0,b0,gamma=reads[0],reads[2],reads[4],reads[-1]
    mat,y=writes[2:]
    # Gamma's DMA may issue before materialization. MTE3_V constrains vector
    # consumers, not launch order of independent MTE2 traffic.
    gate=max(mat['retire'],gamma['retire'])
    phases=dict(dispatch=w0['start'],weights=p0['start']-w0['start'],prefix=b0['start']-p0['start'],
        score_weighted=mat['start']-b0['start'],norm=y['start']-gate,
        materialized_or_gamma=gate-mat['start'],write=y['retire']-y['start'],
        drain=round(s['ca_us']*1650)-y['retire'])
    assert min(phases.values())>=-1,(sh,phases)
    # Conserve original all-core window with explicit removed dependency.
    assert abs(sum(phases.values())-1650*s['ca_us'])<2
    phases['gamma_read']=gamma['retire']-gamma['start']
    return dict(source=str(path.relative_to(R)),trace=s['trace'],trace_sha256=s['trace_sha256'],
        core=c,shape=sh,chip=chip,development=train and chip==DT,ca_us=s['ca_us'],cycles=phases)
def design(field,h,n):
    g=math.ceil(h/64);b=n+1;U=2*h
    # 48.36296 B/clock from the historical 950 MTE2 microsegment.
    # Independent read response intercept is retained in the fixed terms.
    x={'dispatch':[1],'weights':[1,g],'prefix':[1,g],
       'score_weighted':[1,b,g*b],'norm':[1,g],'gamma_read':[1],'write':[1],'drain':[1]}[field]
    transfer={'weights':2,'prefix':2,'score_weighted':n,'gamma_read':1}.get(field,0)*U/48.3629559884
    if field=='write':transfer=U*16.01826/1024
    return x,transfer
def fit():
    rows=[phase_data(*v) for v in inputs()]
    fields=('dispatch','weights','prefix','score_weighted','norm','gamma_read','write','drain')
    fitted={}
    for fld in fields:
        train=[r for r in rows if r['development']];data=[design(fld,r['shape'][1],r['shape'][3]) for r in train]
        x=np.array([a for a,b in data]);y=np.array([r['cycles'][fld]-b for r,(a,b) in zip(train,data)])
        if x.shape[1]==1:coef=[max(0,float(np.median(y)))]
        else:coef=nnls(x,y)[0].tolist()
        fitted[fld]=dict(coefficients=coef,rank=int(np.linalg.matrix_rank(x)))
        for r in rows:
            a,b=design(fld,r['shape'][1],r['shape'][3]);r.setdefault('predicted_reference_cycles',{})[fld]=b+sum(i*j for i,j in zip(a,coef))
    write(O/'phase-calibration.json',dict(roles='v0.19 DT development only. PR and v0.21 are archived sibling-phase replay. No direct 40021 measurements used.',phases=fitted,rows=rows))
    print(len(rows),sum(r['development'] for r in rows),fitted)
if __name__=='__main__':fit()
