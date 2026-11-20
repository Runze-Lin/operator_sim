"""Reconstruct tile-level gamma/x, individual VFs and both stores from complete official windows."""
from collections import defaultdict
import json,sys
from pathlib import Path
from math import ceil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events,digest
from development.validation.rms27.campaign import CHIPS
from development.validation.sku19.campaign import read,write
from operator_model.rms_lane_schedule import _launch_class
from operator_model import load_chip
OUT=ROOT/'results/v0.28/rms';P27=ROOT/'results/v0.27/rms'
KNAME='RmsNorm_normal_all_bf16_high_performance_5000'
SOURCES={'v0.23':ROOT/'results/rms-unknown23','v0.27':P27}
def load_rows(version):
    path=SOURCES[version];b=read(path/('score.json' if version=='v0.27' else 'independent.json'));key='measured' if version=='v0.27' else 'rows'
    return b[key]
def run():
    raw=ROOT/'build/rms28-evidence.json';allrows=[];index=[]
    for vers in SOURCES:
        ref={}
        folder=SOURCES[vers]
        for d in (('verified',) if vers=='v0.23' else ('receipts','recovery')):
            for f in sorted((folder/d).glob('*.json')):
                for r in read(f)['rows']:
                    if r['timing_qualified']:ref[r['stage']['trace']]=r['stage']
        for r in load_rows(vers):
            stage=ref[r['trace']];assert stage['kernel']==KNAME and stage['trace_sha256']==r['trace_sha256'];t,w=r['shape'];sku=r['chip'];chip=load_chip(sku)
            a=r['active_aiv'];h=r['prior_aiv'];stride=ceil(t/chip.vector_cores);d=(ROOT/r['trace']).parent
            pc0=int((d.parent/'dump/pc_start_addr.txt').read_text(),16);origin=stage['first_tick'];hot=r['role'] in ('immediate_repeat','repeat')
            z=dict(version=vers,chip=sku,shape=[t,w],H=h,A=a,hot=hot,ca_us=r['ca_us'],trace=r['trace'],sha=stage['trace_sha256'],lanes=[])
            for lane,core in stage['cores'].items():
                sh=d/lane/'trace.json';assert digest(sh)==core['shard_sha256'];ops=defaultdict(list)
                for v in events(sh):
                    if v['name']!='VF' and not v['name'].startswith('MOV_SRC_TO_DST'):continue
                    pc=int(v['args']['pc_addr'],16)-pc0;type_={0x3e9c:'metadata',0x4878:'g',0x4a48:'x',0x4e64:'s',0x5008:'n',0x4f38:'r',0x5128:'y'}.get(pc)
                    if not type_:assert v['name']=='VF';type_='v'
                    b=round(v['ts']*1650)-origin;e=round((v['ts']+v['dur'])*1650)-origin;de=json.loads(v['args']['detail'])
                    ops[type_].append(dict(b=b,e=e,B=de.get('processed_bytes'),src=de.get('src_addr'),dst=de.get('dst_addr'),pc=hex(pc)))
                for vals in ops.values():vals.sort(key=lambda x:x['b'])
                count=len(ops['r']);assert len(ops['g'])==1 and all(len(ops[k])==count for k in 'xvsny')
                z['lanes'].append(dict(name=lane,arrival=core['first_tick']-origin,end=core['last_tick']-origin,ops=dict(ops)))
            base=min(l['ops']['r'][0]['dst'] for l in z['lanes']);seen=set()
            for l in z['lanes']:
                op=l['ops'];t0=(op['r'][0]['dst']-base)//4;assert t0%stride==0 and t0<t
                block=t0//stride;assert block not in seen;seen.add(block)
                st=('first' if h==0 else _launch_class(block,a,h,ceil(chip.vector_cores/chip.cube_cores)))
                l.update(block=block,state=st,row0=t0);cursor=t0
                for x,rs in zip(op['x'],op['r']):
                    m=rs['B']//4;assert x['B']==w*m*2 and (rs['dst']-base)==cursor*4;cursor+=m
            assert len(seen)==a;allrows.append(z)
            winner=max(z['lanes'],key=lambda l:l['end']);index.append({**{k:v for k,v in z.items() if k!='lanes'},'critical':winner,'lane_count':len(seen)})
    write(raw,allrows);write(OUT/'milestones.json',dict(source='Prior qualified full-ELF device windows used as development',rows=index));print(len(allrows),sum(len(v['lanes'])for v in allrows))
if __name__=='__main__':run()
