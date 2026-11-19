"""Rebuild the chip-wide payload frontier from all existing CA core shards."""
import hashlib
import json
import mmap
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.25/cache'
KEY = b'"name":"MOV_SRC_TO_DST_ALIGNv2"'
D = json.JSONDecoder()
sys.path.insert(0,str(ROOT/'development/validation/pro_prefill23'))
from evidence import endpoint


def load(path):
    return json.loads(path.read_bytes())


def extract(path):
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        start=0
        while (k:=data.find(KEY,start)) != -1:
            pos=data.rfind(b'{"args":{',max(start,k-8192),k)
            if pos < 0: raise ValueError(f'missing event prefix {path} byte {k}')
            e,_=D.raw_decode(data[pos:k+400].decode())
            start=k+len(KEY)
            if e['ph']!='X' or e['dur']<=0:continue
            det=json.loads(e['args']['detail'])
            if det['is_duration_excluded']:continue
            a=e['ts'];b=a+e['dur']
            if [det['src_mem'],det['dst_mem'],det['instr_type']]==['OUT','UB','MTE2']:
                yield 'read', (det['src_addr'],det['processed_bytes'],a,b)
            if [det['src_mem'],det['dst_mem'],det['instr_type']]==['UB','OUT','MTE3']:
                yield 'write', (det['dst_addr'],det['processed_bytes'],a,b)


def tally(events, count):
    groups=[]
    for addr,n,a,b in sorted(events):
        if not groups or addr != groups[-1]['gm_end']:
            groups.append(dict(gm_base=addr,gm_end=addr,first_issue=a,last_retire=b,bytes=0,descriptors=0))
        r=groups[-1];r['gm_end']+=n;r['bytes']+=n;r['descriptors']+=1;r['first_issue']=min(r['first_issue'],a);r['last_retire']=max(r['last_retire'],b)
    matches=[r for r in groups if r['bytes']==count]
    if len(matches)!=1: raise ValueError(f'expected unique contiguous {count}-byte payload; got {groups[:8]}')
    chosen=matches[0]; chosen['window_us']=chosen['last_retire']-chosen['first_issue'];chosen['logical_GB_s']=count/chosen['window_us']/1000
    return chosen,[r for r in groups if r is not chosen]


def measure(receipt):
    s=receipt['stage']
    root=(ROOT / s['trace']).parent
    paths={p.parent.name:p for p in root.glob('core*/trace.json')}
    if set(paths)!=set(s['windows']):raise ValueError('roster mismatch')
    r,w=[],[];fp=hashlib.sha256();origins={}
    for core,path in sorted(paths.items()):
        end=endpoint(path);terminal=max(v['retire'] for v in end.values())/1650
        shift=terminal-s['windows'][core][1];origins[core]=shift
        parts=[(side,(addr,n,a-shift,b-shift)) for side,(addr,n,a,b) in extract(path)]
        fp.update(json.dumps((core,parts),separators=(',',':')).encode())
        for op,e in parts:(r if op=='read' else w).append(e)
    if (max(origins.values())-min(origins.values()))*1650>3:raise ValueError('shard origins disagree')
    cast=s['kernel'].startswith('Cast_')
    n=receipt['shape'][0]*receipt['shape'][1]
    ri,ro=tally(r,n*(2 if cast else 4));wi,wo=tally(w,4*n)
    init=ri['first_issue'];last=ri['last_retire']
    return dict(op='cast_bf16_fp32' if cast else 'sigmoid_fp32',shape=receipt['shape'],chip=receipt.get('chip',receipt.get('sku')),
        raw_trace=s['trace'],aggregate_trace_sha256=s['trace_sha256'],dma_records_sha256=fp.hexdigest(),kernel=s['kernel'],binary_sha256=s['binary_sha256'],
        global_origins_us=origins,origin_check_width_cycles=(max(origins.values())-min(origins.values()))*1650,roster=len(paths),read=ri,write=wi,other_gm_reads=ro,other_gm_writes=wo,ca_us=s['ca_us'],logical_read_lower_bound_GB_s_if_cold=ri['logical_GB_s'],
        conditional_remaining_link_budget_MiB=(max(wi['last_retire'],last)-init)*(1.4e6 if 'PR' in receipt.get('chip',receipt.get('sku')) else 4e6)/2**20-ri['bytes']/2**20)


def samples():
    catalog=load(ROOT/'results/v0.25/coverage/audit.json')['rows']
    for x in catalog:
        if x['op'] not in ('sigmoid_fp32','cast_bf16_fp32') or 'L2 < 输入+输出 ≤ 2L2' not in x['slices']:continue
        for filename in x['origins']:
            data=load(ROOT/filename)
            for rec in (data if isinstance(data,list) else [data]):
                if 'stage' not in rec and isinstance(rec.get('kernel'),dict):rec=dict(rec,stage=rec['kernel'])
                if rec.get('qualified') and rec.get('shape')==x['shape'] and rec.get('chip',rec.get('sku'))==x['chip'] and rec.get('stage',{}).get('trace_sha256') in x['trace_sha256']:
                    yield x['op'],rec
                    break
            else:continue
            break


if __name__=='__main__':
    for op,s in samples():
        dest=OUT/(op+'-'+'x'.join(map(str,s['shape']))+'-'+s.get('chip',s.get('sku'))+'-traffic.json')
        if dest.exists() and 'global_origins_us' in load(dest):continue
        print('reading',dest.name,flush=True)
        result=measure(s);temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
