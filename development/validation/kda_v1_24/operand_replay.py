"""Post-capture CPU sensitivity checks tied to concrete fused V1 release source."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from investigate import REG,OUT,RAW,read,save,probe
from golden import bf

def matrices(tokens,heads,stage):
    aq=np.zeros((heads,tokens,64),'f4');ak=np.zeros_like(aq)
    ti=np.arange(tokens)[:,None];dim=np.arange(128);i=ti*128+dim
    q=bf(((i*7+3)%29-14)/64);key=bf(((i*11+5)%31-15)/128)
    beta=((3+np.arange(tokens)%5)/16).astype('f4')
    for h in range(heads):
        raw=bf(-5+((ti+3*dim+h)%11-5)/8);bias=(((h+dim)%7-3)/32).astype('f4')
        gate=-5/(1+np.exp(-np.exp(np.float32(.125*h))*(raw+bias)))
        for j in range(0,tokens,64):
            n=min(tokens-j,64);g=np.cumsum(gate[j:j+n],axis=0)*np.float32(1/np.log(2))
            q0=q[j:j+n];k0=key[j:j+n];b0=beta[j:j+n,None]
            vals=np.zeros((n,n),'f4');kk=np.zeros_like(vals)
            width=16 if stage=='published_v1_atk' else 32
            for a in range(0,n,width):
                b=min(n,a+width);ref=g[b-1 if stage=='published_v1_atk' else (a+b)//2]
                left=np.exp2(g[a:b]-ref);right=bf(k0[:b]*np.exp2(ref-g[:b]))
                vals[a:b,:b]=(bf(q0[a:b]*left)@right.T)*np.float32(1/np.sqrt(128))
                if stage=='kernel_mid32_beta_in_operand' and heads%2==0:
                    kk[a:b,:b]=bf((k0[a:b]*left)*b0[a:b])@right.T
                else:
                    kk[a:b,:b]=(bf(k0[a:b]*left)@right.T)*b0[a:b]
            aq[h,j:j+n,:n]=bf(np.tril(vals));eye=np.eye(n,dtype='f4')
            ak[h,j:j+n,:n]=bf(solve_triangular(eye+np.tril(kk,-1),eye,lower=True))
    return (aq,ak)

def main():
    impl=next(probe.VENDOR.glob('**/chunk_kda_fwd/arch35/chunk_kda_fwd_prepare.h'))
    text=impl.read_text()
    for token in ('KDA_SAFE_SCORE_REF_BC = 32','rowBegin + rowCount / 2','scoreCanIncludeBeta && HV_ % KDA_SCORE_LANES == 0','if constexpr (SCALE_SCORE_W)'):
        if token not in text:raise ValueError('published source missing '+token)
    rows=[]
    for c in read(REG)['cases']:
        p=RAW/c['id'];t,h=c['shape'][1:3]
        if not all((p/(n+'.bf16')).exists() for n in ('aqk','akk')):continue
        actual=[(np.fromfile(p/(n+'.bf16'),'<u2').astype('<u4')<<16).view('<f4').reshape(h,t,64) for n in ('aqk','akk')]
        by={}
        for strategy in ('published_v1_atk','kernel_mid32','kernel_mid32_beta_in_operand'):
            ref=matrices(t,h,strategy);by[strategy]={}
            for name,a,e in zip(('aqk','akk'),actual,ref):
                if strategy=='published_v1_atk':
                    frozen=np.fromfile(RAW/'oracles'/f'{t}-{h}'/(name+'.bin'),'<f8').reshape(a.shape)
                    if not np.array_equal(e,frozen):raise ValueError('replay differs from pre-capture oracle')
                delta=abs(a-e);bad=delta>2e-6+abs(e)/64
                idx=np.unravel_index(delta.argmax(),a.shape)
                by[strategy][name]=dict(bad=int(bad.sum()),unmatched_bits=int((a!=e).sum()),count=int(a.size),
                    max_abs=float(delta[idx]),worst_index=list(map(int,idx)),observed=float(a[idx]),expected=float(e[idx]),
                    first_failed_coordinates=[dict(index=list(map(int,k)),actual=float(a[tuple(k)]),
                        reference=float(e[tuple(k)]),abs_error=float(delta[tuple(k)]),
                        allowed=float(2e-6+abs(e[tuple(k)])/64)) for k in np.argwhere(bad)[:4]])
        rows.append(dict(id=c['id'],rows=by))
    save(OUT/'operand-replay.json',dict(role='retrospective diagnostic only; original frozen verdict retained',
        source_sha256=probe.digest(impl),source=probe.rel(impl),source_lines=[82,387,510,1100,1726,3389,4454],
        interpretations=['ATK: 16-row last-row shift, beta after score GEMM.','arch35: 32-row midpoint shift.','arch35 paired-head specialization: beta is multiplied into key before the BF16 cast and score MMAD.'],cases=rows))
if __name__=='__main__':main()
