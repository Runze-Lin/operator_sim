"""Fit instruction and DMA interval prices, with one observation per launch, lane state and tile size."""
from collections import defaultdict
from math import ceil
from statistics import median
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import lsq_linear
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.rms_resource import READ_PER_KIB,FIRST_PAYLOAD_RELEASE
from development.validation.rms28.extract import OUT
from development.validation.sku19.campaign import read,write

def features(width,rows):
    # 64 FP32 values per 256-byte vector register; the installed pair-load loop handles 128.
    vec_groups=ceil(width/128);tree=(ceil(width/64)-1).bit_length()
    return [1,rows,rows*vec_groups,rows*tree]

def linear(X):
    # Include each launch once per observed lane class and distinct full/tail tile size.
    X=np.array(X,float);a=X[:,:-1];b=X[:,-1]
    if len(b)<a.shape[1]+1 or all(np.ptp(a[:,i])==0 for i in range(1,a.shape[1])):
        return [max(0.,float(np.median(b)))]+[0.]*(a.shape[1]-1)
    # Huber influence, with the residual scale estimated from the measured stage rather than knob tuned to total time.
    coef=lsq_linear(a,b,bounds=(0,np.inf)).x; weights=np.ones(len(b))
    for _ in range(6):
        err=b-a@coef;mad=max(1.,1.4826*float(np.median(np.abs(err-np.median(err)))))
        weights=np.minimum(1,1.345*mad/np.maximum(abs(err),1e-12))
        z=np.sqrt(weights);coef=lsq_linear(a*z[:,None],b*z,bounds=(0,np.inf)).x
    return [float(x) for x in coef]

def observations(records):
    out=[]
    for row in records:
        a=row['A'];t,w=row['shape'];sku=row['chip'];chip=load_chip(sku);h=row['H']
        by=defaultdict(list)
        def sample(k,x,y):by[k,tuple(x)].append(y)
        for lane in row['lanes']:
            st='repeat' if row['hot'] else lane['state'];o=lane['ops'];g=o['g'][0]
            first_payload=(chip.details.get('rms_first_payload_cycles') or FIRST_PAYLOAD_RELEASE)
            sample('gamma_'+st,[1,a-1],g['b']-(first_payload-FIRST_PAYLOAD_RELEASE if st=='first' else 0))
            sample('gamma_spread_'+st,[1,a-1],g['b'])
            sample('x_issue',[1],o['x'][0]['b']-g['b'])
            sample('gamma_response_'+st,[1,g['B']/1024],g['e']-g['b'])
            for i,(x,v,s,n,r,y) in enumerate(zip(*(o[k] for k in 'xvsnry'))):
                count=r['B']//4
                # All participants request the first block; a short last lane requests fewer rows.
                global_bytes=sum(z['ops']['x'][i]['B'] for z in row['lanes'] if len(z['ops']['x'])>i)
                bus=global_bytes*chip.frequency_ghz/chip.hbm_bandwidth_gbs if not row['hot'] else 0.
                amount=x['B']/1024
                sample('x_response_'+st,[1,amount,bus],x['e']-x['b'])
                if i==0:sample('read_to_vf',[1],v['b']-max(g['e'],x['e']))
                if i:
                    sample('second_x_'+st,[1,o['x'][i-1]['B']/1024],x['b']-o['x'][i-1]['b'])
                    sample('vector_reissue',[1],o['n'][i-1]['e']-v['b'])
                f=features(w,count)
                sample('reduce_to_rstd_vf',f,s['b']-v['b'])
                sample('rstd_to_normalize_issue',[1,count],n['b']-s['b'])
                sample('rstd_vf_response',[1,count],s['e']-s['b'])
                sample('normalize_response',f,n['e']-n['b'])
                sample('issue_rstd',[1],r['b']-s['e'])
                sample('issue_y',[1],y['b']-n['e'])
                sample('y_service_'+st,[1,y['B']/1024],y['e']-y['b'])
                sample('rstd_solo_'+st,[1],r['e']-r['b'])
        for (k,x),q in by.items():
            fn=(lambda z:max(z)-min(z)) if k.startswith('gamma_spread_') else min if k.startswith(('y_service_','rstd_solo_')) else median
            out.append(dict(version=row['version'],sku=sku,shape=row['shape'],trace=row['trace'],key=k,x=list(x),y=fn(q),count=len(q)))
    return out

def fit(points):
    sets=defaultdict(list)
    for r in points:sets[r['key']].append([*r['x'],r['y']])
    prices={}
    for k,rows in sets.items():
        if k.startswith('gamma_') and 'response' not in k and len({r[1]for r in rows})<4:
            prices[k]=[median(r[-1] for r in rows),0.]
        else:prices[k]=linear(rows)
    return dict(prices=prices,n={k:len(v)for k,v in sets.items()})

if __name__=='__main__':
    points=observations(read(ROOT/'build/rms28-evidence.json'));write(OUT/'stage-samples.json',points);write(OUT/'prices.json',fit(points))
