"""FLA fused V1 CPU ATK reference for the mixed-sign BF16 Kimi fixture."""
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular


def bf(x):
    u=np.asarray(x,np.float32).view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def make(t,h,folder):
    folder.mkdir(parents=True,exist_ok=True);x=np.arange(t)[:,None]*128+np.arange(128);dtype=np.float32
    q=bf(((7*x+3)%29-14)/64);k=bf(((11*x+5)%31-15)/128);v=bf(((5*x+7)%23-11)/2048)
    beta=((3+np.arange(t)%5)/16).astype(dtype);tt=np.arange(t)[:,None];d=np.arange(128)
    out=np.zeros((t,h,128),dtype);final=np.zeros((h,128,128),dtype)
    aq=np.zeros((h,t,64),dtype);ak=np.zeros_like(aq)
    for head in range(h):
        raw=bf(-5+((tt+3*d+head)%11-5)/8);bias=(((head+d)%7-3)/32).astype(dtype)
        gate=-5/(1+np.exp(-np.exp(dtype(.125*head))*(raw+bias)))
        previous=((np.arange(128*128)*13+1)%19-9).reshape(128,128).astype(dtype)/8192
        for j in range(0,t,64):
            n=min(64,t-j);z=slice(j,j+n);bs=beta[z,None];prefix=np.cumsum(gate[z],axis=0)*dtype(1/np.log(2))
            qb=q[z];kb=k[z];vb=v[z]
            scores=np.zeros((n,n),dtype);kk=np.zeros_like(scores)
            # The fused V1 ATK uses the LAST row of each band, distinct from V2's midpoint.
            for a in range(0,n,16):
                b=min(n,a+16);shift=prefix[b-1]
                lo=np.exp2(prefix[a:b]-shift);hi=np.exp2(shift-prefix[:b]);right=bf(kb[:b]*hi)
                scores[a:b,:b]=bf(qb[a:b]*lo)@right.T*dtype(1/np.sqrt(128))
                kk[a:b,:b]=bf(kb[a:b]*lo)@right.T
            scores=np.tril(scores);eye=np.eye(n,dtype=dtype)
            inv=solve_triangular(eye+np.tril(kk*bs,-1),eye,lower=True)
            iq=bf(inv);sq=bf(scores);eg=np.exp2(prefix)
            w=bf(iq@bf(kb*bs*eg));u=bf(iq@bf(vb*bs))
            kg=bf(kb*np.exp2(prefix[-1]-prefix));qg=bf(qb*eg)
            prev=bf(previous);vn=bf(u-w@prev)
            out[z,head]=bf((qg@prev)*dtype(1/np.sqrt(128))+sq@vn)
            previous=np.exp2(prefix[-1])[:,None]*previous+kg.T@vn
            aq[head,z,:n]=sq;ak[head,z,:n]=iq
        final[head]=previous.T
    for n,a in [('attention',out),('aqk',aq),('akk',ak),('state_vk',final)]:
        a.astype('<f8').tofile(folder/(n+'.bin'))

if __name__=='__main__':make(int(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]))
