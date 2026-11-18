"""Measure the 56-tile PR BF16 matmul using independent MMAD/FIX descriptors."""
from collections import Counter,defaultdict
import json
from campaign import *
def main():
 r=next(r for r in load(OUT/'holdout.json') if r['id']=='matmul_1536_2048_16')
 tr=ROOT/r['pr']['trace'];assert sha(tr)==r['pr']['trace_sha256']
 M,N,K=r['shape'];cores=defaultdict(list);fix=[];t0=None
 for e in load(tr)['traceEvents']:
  if e['ph']!='X':continue
  t0=min(t0 if t0 is not None else e['ts'],e['ts'])
  if e['name'] not in ('MMAD','FIX_L0C_TO_DST'):continue
  d=json.loads(e['args']['detail'])
  if e['name']=='MMAD':cores[e['pid']].append(dict(m=d['m'],n=d['n'],k=d['k'],init=d['matrix_c_init_val'],ts=e['ts'],dur=e['dur']))
  else:fix.append((d['dst_addr'],d['m'],d['n']))
 base=min(a for a,_,_ in fix);outputs=[(a-base)//2 for a,m,n in fix];row_intervals=defaultdict(list)
 for i,(a,m,n) in zip(outputs,fix):
  y,x=divmod(i,N);assert y+m<=M and x+n<=N and (a-base)%2==0
  for row in range(y,y+m):row_intervals[row].append((x,x+n))
 contiguous=all(sorted(row_intervals[y])[0][0]==0 and sorted(row_intervals[y])[-1][1]==N and all(a[1]==b[0]for a,b in zip(sorted(row_intervals[y]),sorted(row_intervals[y])[1:])) for y in range(M))
 def hist(xs):return dict(Counter(xs))
 for cs in cores.values():
  cs.sort(key=lambda z:z['ts'])
  for x in cs:x['ts']-=t0
 flat=[d for c in cores.values() for d in c]
 result=dict(shape=r['shape'],trace=str(tr.relative_to(ROOT)),trace_sha256=sha(tr),predicted_plan=r['projection']['target_plan'],pr_ca_us=r['pr_ca_us'],resource_us=r['pr_us'],measured_mmad_per_core=cores,mmad_per_core_count_histogram=hist(map(str,map(len,cores.values()))),mmad_mnk_histogram=hist(f"{x['m']}x{x['n']}x{x['k']}" for x in flat),mmad_init_histogram=hist(str(x['init']) for x in flat),fix_shape_histogram=hist(f'{m}x{n}' for _,m,n in fix),unique_output_addresses=len(set(outputs)),output_fix_count=len(fix),all_fix_rectangles_cover_mn_once=contiguous,all_mmads_cover_full_k=all(x['k']==K for x in flat),sum_mmad_ops=sum(2*x['m']*x['n']*x['k'] for x in flat),logical_ops=2*M*N*K)
 write(OUT/'1536-multiwave.json',result);print({k:v for k,v in result.items() if 'predicted_plan'!=k and k!='measured_mmad_per_core'})
if __name__=='__main__':main()
