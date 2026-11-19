"""Test a source-level 16-lane masked-tail explanation against archived aggregate golden counts."""
import json
import re
from struct import pack, unpack

from extract import OUT,ROOT,read,sha


def bf16(x):
    u,=unpack('=I',pack('=f',x))
    return (u+0x7fff+((u>>16)&1))>>16


def diagnose():
    tokens,hidden,w=7,4112,16
    per_stream=[0]*4;ties=[]
    def data(index,offset):return ((index+offset)%17-8)/16
    for row in range(tokens):
        post=lambda j: ((4*row+j)%7-3)/8
        comb=lambda i,j: ((16*row+4*i+j)%11-5)/16
        x=lambda k: data(row*hidden+4096+k,7)
        residual=lambda i,k: data((4*row+i)*hidden+4096+k,4) if i<4 else 0.
        for stream in range(4):
            for k in range(w):
                correct=x(k)*post(stream)+sum(comb(i,stream)*residual(i,k) for i in range(4))
                # First stream got 64 lanes in a layout with only 16 valid entries per stream.
                # Its leaked lanes see the next packed residual streams, then unfilled UB.
                candidate=(x(k) if stream==0 else 0.)*post(0)+sum(comb(i,0)*residual(i+stream,k) for i in range(4))
                if bf16(correct)!=bf16(candidate):
                    per_stream[stream]+=1
                elif stream:
                    ties.append(dict(row=row,stream=stream,d=4096+k,bf16=bf16(correct)))
    observations=[]
    for sku in ('Ascend950DT_9582','Ascend950PR_957b'):
        file=OUT/f'holdout-{sku}-t7-d4112.json';r=read(file)
        log=ROOT/r['raw']/'run.log';txt=log.read_text()
        n,bad=map(int,re.search(r'correctness name=output count=(\d+) bad=(\d+)',txt).groups())
        assert n==tokens*4*hidden and not r['qualified'] and 'SMOKE_FAIL' in txt
        observations.append(dict(sku=sku,full_checked=n,actual_all_streams_bad=bad,source_sha256=sha(log),receipt=file.name))
    return dict(shape=[1,tokens,4,hidden],partial_tile=w,formula='UpdateMask<float>(xDealNum) advances a shared remaining-lane counter initialized at 4*w, so for w=16 the predicted valid-lane masks of the four outer-loop streams are [64,0,0,0].',
                predicted_lanes_per_stream=[64,0,0,0],assumption='Extra lanes in the second x and residual UB slots which were never populated are modeled as zero; valid residual stream blocks are contiguous.',
                predicted_bad_by_stream=per_stream,predicted_bad_total=sum(per_stream),accidental_matches=ties,
                actual_bad_by_stream=None,measured=observations,interpretation='Both hosts logged full-tensor mismatch counts only. Per-lane values and extra UB contents were not dumped and remain inferred; matching the two aggregate counts cannot establish the per-stream identity uniquely.')


if __name__=='__main__':
    out=diagnose()
    (OUT/'failed-tail-mask.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out['predicted_bad_by_stream'],out['measured'])
