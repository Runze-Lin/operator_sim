"""CPU reproduction of the published FLA prepare golden and the math-only comparator."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular


def bf16(x):
    a=np.asarray(x,dtype='<f4').view('<u4')
    hi=(a+np.uint32(0x7fff)+((a>>16)&1))&np.uint32(0xffff0000)
    return hi.view('<f4').astype(np.float64)


def prepare(tokens,heads):
    aq=np.zeros((heads,tokens,64));ak=np.zeros_like(aq)
    math_q=np.zeros_like(aq);math_k=np.zeros_like(aq)
    t=np.arange(tokens)[:,None]; d=np.arange(128);i=t*128+d
    q=bf16(((i*7+3)%29-14)/64);k=bf16(((i*11+5)%31-15)/128)
    beta=(3+t[:,0]%5)/16
    for h in range(heads):
        gate=bf16(-5+((t+3*d+h)%11-5)/8)
        gate=-5/(1+np.exp(-np.exp(h*.125)*(gate+((h+d)%7-3)/32)))
        for j in range(0,tokens,64):
            n=min(64,tokens-j); qs,ks=q[j:j+n],k[j:j+n];be=beta[j:j+n,None]
            # vLLM direct V2 uses base-2 inside the published Prepare implementation.
            g=np.cumsum(gate[j:j+n],axis=0)/np.log(2)
            tri=np.zeros((n,n)); exact=np.zeros_like(tri)
            for a in range(0,n,16):
                b=min(a+16,n);ref=g[(a+b)//2]
                left=np.exp2(np.clip(g[a:b]-ref,-126,120));right=np.exp2(np.clip(ref-g[:b],-126,120))
                minus=bf16(ks[:b]*right)
                aq[h,j+a:j+b,:b]=bf16((bf16(qs[a:b]*left)@minus.T)/np.sqrt(128))
                tri[a:b,:b]=be[a:b]*(bf16(ks[a:b]*left)@minus.T)
            aq[h,j:j+n,:n]=np.tril(aq[h,j:j+n,:n])
            for row in range(n):
                decayed=ks[:row+1]*np.exp2(g[row]-g[:row+1])
                math_q[h,j+row,:row+1]=np.sum(decayed*qs[row],axis=1)/np.sqrt(128)
                exact[row,:row+1]=np.sum(decayed*ks[row],axis=1)*be[row,0]
            # Exact FP64 lower triangular inversion is numerically equivalent to the
            # official pair of 32x32 triangular inverses with lower-left block product.
            ak[h,j:j+n,:n]=bf16(solve_triangular(np.eye(n)+np.tril(tri,-1),np.eye(n),lower=True))
            math_k[h,j:j+n,:n]=solve_triangular(np.eye(n)+np.tril(exact,-1),np.eye(n),lower=True)
    return (aq,ak),(math_q,math_k)


def diff(actual,expect):
    err=np.abs(actual-expect);limit=2e-6+np.abs(expect)/64;idx=np.unravel_index(np.argmax(err),err.shape)
    rel=err/np.maximum(abs(expect),np.finfo('d').tiny)
    return dict(count=int(err.size),bad=int((err>limit).sum()),max_abs=float(err[idx]),worst_abs_at=list(map(int,idx)),
                observed=float(actual[idx]),expected=float(expect[idx]),max_rel=float(rel.max()),mean_abs=float(err.mean()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tokens',type=int);p.add_argument('heads',type=int);p.add_argument('output',type=Path);args=p.parse_args()
    actual,mathematical=prepare(args.tokens,args.heads);args.output.mkdir(parents=True,exist_ok=True)
    comparison={}
    for label,a,b in zip(('aqk','akk'),actual,mathematical):
        a.astype('<f8').tofile(args.output/(label+'.bin'));comparison[label]=diff(a,b)
    (args.output/'math-vs-published.json').write_text(json.dumps(comparison,indent=2)+'\n')
