"""Predeclare one matched-backend DT application per extension host route."""
from concurrent.futures import ThreadPoolExecutor
from campaign_collect import *
def main():
 b=read(OUT/'extension-freeze.json')
 jobs=[j for j in b['jobs']if j['rank']==0 and j['op']!='kda_chunk']
 reg=OUT/'paired-control-budget.json'
 if not reg.exists():write(reg,dict(utc=utc(),selection='first predeclared case for each host route; independent of accuracy; matched application and exact argv sequence',ids=[j['id'] for j in jobs],workers=2,max_cases=9))
 def one(j):
  d=RAW/'extension'/(j['id']+'_'+DT)
  if not(d/'cost.json').exists():capture(j,DT,d,b)
  try:
   r=qualify(j,DT,d);write(d/'strict.json',r);print('CONTROL',j['id'],'OK',flush=True)
  except (ValueError,IndexError)as e:print('CONTROL',j['id'],repr(e),flush=True)
 with ThreadPoolExecutor(2)as ex:list(ex.map(one,jobs))
if __name__=='__main__':main()
