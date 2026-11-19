"""BF16 ND: the ASW final round can be distributed among idle physical cores."""
from collections import Counter
from math import ceil


def _parts(n,step):
    whole,last=divmod(n,step)
    return [(size,count) for size,count in ((step,whole),(last,1)) if size and count]


def dispatch(shape,plan):
    m,n,_=shape;am,an,_=plan['tile'];active=plan['active_aic'];jobs=plan['output_tiles']
    gm,gn=ceil(m/am),ceil(n/an); allh=Counter({(x,y):a*b for x,a in _parts(m,am) for y,b in _parts(n,an)})
    # Only the final wave is enumerated. All earlier jobs reduce to a rectangle histogram.
    tail=(jobs-1)%active+1; smb=snb=1
    if jobs>active and tail<active:
        while (smb+1)*snb*tail<=active:
            smb+=1
            if smb*(snb+1)*tail<=active and ceil(an/snb)*2>128:snb+=1
        # Runtime clips host split counts against the real shape of the grid's last parent.
        px=(m-1)%am+1;py=(n-1)%an+1
        smb=ceil(px/ceil(px/smb));snb=ceil(py/ceil(py/snb))
    width=min(4,gm);main=gm//width-1
    lasth=Counter();parents=Counter()
    for i in range(jobs-tail,jobs):
        row=min(i//(gn*width),main)
        w=width if row<main else gm-main*width
        ix=i-row*width*gn;x=row*width+ix%w;y=ix//w
        if row%2:y=gn-1-y
        p,q=min(am,m-x*am),min(an,n-y*an)
        parents[p,q]+=1
        for mm,u in _parts(p,ceil(p/smb)):
            for nn,v in _parts(q,ceil(q/snb)):
                lasth[mm,nn]+=u*v
    return (allh-parents,lasth,(smb,snb))
