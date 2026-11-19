"""Retain the frozen predictions and add the physical DCCI preceding a MIX END."""
import hashlib
import json
import mmap
from pathlib import Path
import sys

import campaign

OUT=campaign.OUT
DEC=json.JSONDecoder()
original=campaign.endpoint
ADD=OUT/'endpoint-addendum.json'


def terminal(path):
    marks=original(path)
    expected=int(marks['END']['pc'],16)-4
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as text:
        key=b'"name":"DCCI"';k=text.rfind(key)
        if k>=0:
            i=text.rfind(b'{"args":{',max(0,k-4096),k)
            if i<0:raise ValueError('malformed DCCI event')
            e,_=DEC.raw_decode(text[i:k+512].decode())
            if e['ph']=='X' and e['name']=='DCCI' and int(e['args']['pc_addr'],16)==expected:
                marks['DCCI']=dict(retire=round(1650*(e['ts']+e['dur'])),pc=e['args']['pc_addr'])
    return marks


def register():
    if ADD.exists() or any(p.is_dir() for p in campaign.RAW.iterdir()):raise ValueError('prior attempt')
    controls=[]
    for sku in (campaign.DT,campaign.PR):
        for t in (20,56,80):
            fn=f'mhc_pre_sinkhorn_{t}_4_4096_{sku}.json'
            path=campaign.ROOT/'results/v0.22/recovered'/fn
            if path.exists():
                o=campaign.load(path)['observation']
                wins={k:[(a-o['first_tick'])/1650,(b-o['first_tick'])/1650]for k,(a,b) in o['windows_ticks'].items()}
            else:
                path=campaign.ROOT/'results/v0.21/fusion/receipts'/fn;o=campaign.load(path)['kernel'];wins=o['windows']
            base=(campaign.ROOT/o['trace']).parent
            spreads=[]
            for method in (original,terminal):
                offsets=[max(x['retire'] for x in method(base/c/'trace.json').values())-w[1]*1650 for c,w in wins.items()]
                spreads.append(max(offsets)-min(offsets))
            assert spreads[1]<.1
            controls.append(dict(shape=[t,4,4096],sku=sku,receipt=str(path.relative_to(campaign.ROOT)),cores=len(wins),
                                 original_spread_cycles=spreads[0],verified_spread_cycles=spreads[1]))
    campaign.save(ADD,dict(utc=campaign.now(),file_sha256=campaign.sha(Path(__file__)),original_freeze_sha256=campaign.sha(campaign.REG),controls=controls,
          correction='The generic Cast checker uses DCI. The fused AIC executes DCCI at END PC-4, whose actual retirement can follow END. Include that real record, retain END and the original <=3 cycle limit. No observation for the four new launches was available.'))


def run():
    add=campaign.load(ADD)
    if campaign.sha(Path(__file__))!=add['file_sha256'] or campaign.sha(campaign.REG)!=add['original_freeze_sha256']:raise ValueError('freeze mismatch')
    campaign.endpoint=terminal
    campaign.run()

if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
