"""Closest archived pair is context, never a substitute for an unmeasured target."""
from pathlib import Path
import json, math
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[3];O=ROOT/'results/v0.32/large'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
def rd(p):return json.loads((ROOT/p).read_bytes())
def amount(op,s):
 if op.startswith(('kda','fla_kda')) and len(s)==6:return s[0]*s[1]*s[3]*s[4]*s[5]
 if op in ('moe_ffn','grouped_matmul_swiglu_quant_v2','gmm2') and len(s)==4:return math.prod(s[:3])
 return math.prod(s)
def main():
 catalog=rd('results/v0.32/large/coverage.json');pairs=catalog['complete_same_shape_pairs_by_op'];lookup=defaultdict(dict)
 def add(op,s,attrs,sku,ca,source):
  if sku not in (DT,PR) or ca is None:return
  key=(op,tuple(s),json.dumps(attrs,sort_keys=True,separators=(',',':')))
  # Retain first of repeated archived calls; don't claim a new statistical holdout or mean.
  lookup[key].setdefault(sku,dict(ca_us=ca,source=source))
 for r in rd('results/v0.31/replay.json')['rows']:
  add(r['op'],r['shape'],r['attrs'],r['chip'],r['ca_us'],'results/v0.31/replay.json')
 for r in rd('results/v0.19/sku/final-records.json'):
  add(r['op'],r['shape'],r['attrs'],r['chip'],r['ca_us'],'results/v0.19/sku/final-records.json')
 for r in rd('results/v0.29/large/vector/archive.json'):
  if r['protocol']=='instruction':add(r['op'],r['shape'],r['attrs'],r['chip'],r['ca_us'],r['source'])
 for p in rd('results/v0.32/large/archive.json')['rows']:
  add('mhc_sinkhorn',p['shape'],dict(iterations=p['iterations'],history='ordered v0.26 wide stream'),p['chip'],p['ca_us'],p['old_receipt'])
 rows=[];lines=['# 各路线的归档参照与待测大形','', '左列只在给定资料包内选择同 geometry、相同属性且两端都合格的代表样本（从归档按工作规模挑最大的一项）；大形目标是另一列，不把控制例当作达标。H 作为输入的样本需相同属性；Sinkhorn 的有序流整段相同、实际 H 由各芯片核数产生。单独列固定迭代数避免把 4 次的循环当成 20 次。只有右侧链接汇总表中的同合同大形完整窗计入大形覆盖。','', '| 路线 | 旧库中有双端完整窗的参照 | 950DT / 950PR CA (µs) | 登记的较大或目标输入 |','|---|---|---:|---|']
 for op in sorted(pairs):
  keys=[k for k,v in lookup.items() if k[0]==op and set(v)=={DT,PR}]
  chosen=max(keys,key=lambda k:amount(op,k[1]),default=None)
  new=max((r for r in catalog['rows'] if r['op']==op),key=lambda r:amount(op,r['shape']))
  entry=dict(op=op,target_shape=new['shape'],target_contract=new['contract'],target_source=new['source'],archive=None)
  if chosen:
   o,sh,attrs=chosen;v=lookup[chosen];entry['archive']=dict(shape=sh,attrs=json.loads(attrs),by_chip=v,role='archive context; not new or presumed deployment route')
   s='['+','.join(map(str,sh))+']';t=f"{v[DT]['ca_us']:.3f} / {v[PR]['ca_us']:.3f}"
  else:s=t='N/A'
  lines.append(f'| {op} | {s} | {t} | {new["shape"]} |');rows.append(entry)
 lines+=['','筛选指标仅用来给同一个算子的资料排序：KDA 按 B·S·Hv·K·V；MoE 按 M·K·F，不把专家数重复乘入 token；其余按接口维数乘积。它不是工时预测，也不是严格嵌套的精度/负载证明。不同路由、宽度、dtype 或冷热条件下的设备时间不能直接缩放。例如 Addmm 的旧数不能替代 262144 行带 bias 的投影，12 本地头的 KDA 不能直接用 1 头时间相乘。','', '[全路线大形、协议、角色、误差和实时回执](report.md) · [逐路线 shape、属性与证据来源 JSON](scale-ladder.json)','']
 (O/'scale-ladder.md').write_text('\n'.join(lines));(O/'scale-ladder.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
