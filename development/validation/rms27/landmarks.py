"""Recover the eight dispatched operations from each single-UB AIV in the previous paired series."""
from hashlib import sha256
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_ordered_verify.verify import events,digest
from development.validation.sku19.campaign import read,write
OUT=ROOT/'results/v0.27/rms';OLD=ROOT/'results/rms-unknown23'
def extract():
    measures=[]
    for path in sorted((OLD/'verified').glob('*.json')):
        data=read(path)
        for r in data['rows']:
            if not r['timing_qualified'] or r['role']=='prefix' or r['process']=='multi-UB':continue
            s=r['stage'];folder=(ROOT/s['trace']).parent;base=int((folder.parent/'dump/pc_start_addr.txt').read_text(),16)
            z=dict(chip=r['chip'],shape=r['shape'],role=r['role'],process=r['process'],H=r['prior_aiv'],A=r['active_aiv'],ca_us=s['ca_us'],trace=s['trace'],sha=s['trace_sha256'],lanes=[])
            for lane,q in s['cores'].items():
                f=folder/lane/'trace.json';assert digest(f)==q['shard_sha256']
                op={}
                for ev in events(f):
                    if ev['name']!='VF' and not ev['name'].startswith('MOV_SRC_TO_DST'):continue
                    pc=int(ev['args']['pc_addr'],16)-base
                    name={0x3e9c:'metadata',0x4878:'gamma',0x4a48:'x',0x4e64:'rstd_vf',0x5008:'norm_vf',0x4f38:'rstd',0x5128:'y'}.get(pc)
                    if not name:
                        assert ev['name']=='VF';name='reduce'
                    assert name not in op
                    desc=json.loads(ev['args']['detail']);b=round(ev['ts']*1650)-s['first_tick'];e=round((ev['ts']+ev['dur'])*1650)-s['first_tick']
                    op[name]=dict(b=b,e=e,bytes=desc.get('processed_bytes'),dst=desc.get('dst_addr'),pc=hex(pc))
                assert len(op)==8
                z['lanes'].append(dict(lane=lane,arrival=q['first_tick']-s['first_tick'],end=q['last_tick']-s['first_tick'],ops=op))
            measures.append(z)
    write(OUT/'landmarks.json',dict(source='v0.23 all-GOLD, installed-ELF complete matched windows, one UB tile, cycles from all-device first instruction',rows=measures))
if __name__=='__main__':extract()
