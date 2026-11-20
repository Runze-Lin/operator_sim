"""Read selected closed DT HcPost shards; verify all row sizes before comparing DMA/VF timing."""
from collections import defaultdict
from hashlib import sha256
import json
import mmap
from pathlib import Path
import re
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model.hc_post_timing import _schedule, _parameters, _dma_prices

D=ROOT/'runs/specialized31/post/vllm_hc_post-Ascend950DT_9582/profile'
OUT=ROOT/'results/v0.32/post'
def load(p):return json.loads(p.read_bytes())
def main():
    lanes=[]
    for idx in (0,7,15,23,31):
        for sub in (0,1):
            path,=D.glob(f'*/device0/*/0/simulator/core{idx}.veccore{sub}/trace.json')
            events=defaultdict(list)
            with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
                # Read exact Chrome X event objects of both DMA queues and vector-function dispatch.
                for match in re.finditer(rb'"name":"VF","ph":"X"|"ph":"X","pid":(?:60|70),',raw):
                    a=raw.rfind(b'{"args":',max(0,match.start()-3000),match.start())
                    b=raw.find(b'}',match.end())+1
                    ev=json.loads(raw[a:b])
                    q=ev['pid']; detail=json.loads(ev['args']['detail'])
                    tick=round(ev['ts']*1650)
                    stop=round((ev['ts']+ev['dur'])*1650)
                    if ev['name']=='VF':
                        events['vf'].append([tick,stop,0]);continue
                    if (detail.get('src_mem'),detail.get('dst_mem')) not in (('OUT','UB'),('UB','OUT')):continue
                    if detail.get('processed_bytes',0)>0:events[q].append([tick,stop,detail['processed_bytes']])
                digest=sha256(raw).hexdigest()
            reads=sorted(events[60]);writes=sorted(events[70]);vfs=sorted(events['vf'])
            assert [e[2] for e in reads]==[96]+[16,64,8192,32768,2048,8192]*16
            assert [e[2] for e in writes]==[32768,8192]*16 and len(vfs)==64
            origin=reads[0][0]
            rr=[]
            for i in range(16):
                post,comb,x0,r0,x1,r1=reads[1+i*6:7+i*6]
                w0,w1=writes[2*i:2*i+2]
                v0,c0,v1,c1=vfs[4*i:4*i+4]
                rr.append({k:[e[0]-origin,e[1]-origin] for k,e in dict(coeff=post,comb=comb,x0=x0,r0=r0,x1=x1,r1=r1,write0=w0,write1=w1,v0=v0,c0=c0,v1=v1,c1=c1).items()})
            lanes.append(dict(lane=path.parent.name,shard=str(path.relative_to(ROOT)),sha256=digest,
                meta_issue_tick=origin,rows=rr))
    chip=load_chip(); p=_parameters()['prices'];dma=_dma_prices(chip,p)
    old=[_schedule(i,[4096,1024],63,p,dma,1.) for i in range(1,18)]
    rh=old[-1]['final_write_issue_tick']-old[-2]['final_write_issue_tick']
    summary={}
    for field in ('coeff','x0','r0','write0','v0','c0','x1','r1','write1','v1','c1'):
        rr=[r for l in lanes for r in l['rows'][2:]]
        gaps=[b[field][0]-a[field][0] for l in lanes for a,b in zip(l['rows'][1:],l['rows'][2:])]
        durs=[r[field][1]-r[field][0] for r in rr]
        summary[field]=dict(steady_row_interval_p50=median(gaps),service_p50=median(durs),service_range=[min(durs),max(durs)])
    data=dict(scope='DT development, ten selected closed lane shards (two lanes each at 0,7,15,23,31); independent PR has not been read. All ticks at 1.65 GHz. Window and qualification supplied separately by recovery.',
              v031_public_steady_ii= rh, summary=summary,lanes=lanes)
    (OUT/'hc-dt-rows.json').write_text(json.dumps(data,indent=2)+'\n')
    print('current_II',rh,'actual',summary)
    for l in lanes[::2]:print(l['lane'],[(r['write0'][0],r['write1'][0]) for r in l['rows'][:3]],
              l['rows'][-1]['write1'][1]-l['rows'][0]['write1'][1])
if __name__=='__main__':main()
