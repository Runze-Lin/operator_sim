"""Paired non-FA VF prices. Optional audit extracts first-tile native evidence."""
import argparse
import csv
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path
from operator_model import estimate, load_chip
from operator_model.operators.vector.instruction_classes import tile_service
from operator_model.operators.quant.token_fp8 import coefficients

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/v2.2'
OPS = ('cast_bf16_fp32','sigmoid_fp32','quant_token_fp8')


def extract():
    stages = []
    for p in sorted((ROOT/'runs/vector21').glob('*/profile/*/simulator/core0.veccore0/trace.json')):
        tag = p.parents[4].name
        op = next((name for name in OPS[:2] if tag.startswith(name + '_')), None)
        if not op:
            continue
        m,n,sku,code = tag[len(op)+1:].split('_')
        chip = sku+'_'+code
        data = p.read_bytes()
        ev = sorted((e for e in json.loads(data)['traceEvents'] if e.get('ph')=='X' and e['dur']>0),
                    key=lambda e:e['ts'])
        dmas = [(e,json.loads(e['args']['detail'])) for e in ev if e['name']=='MOV_SRC_TO_DST_ALIGNv2']
        reads = [(e,d['processed_bytes']) for e,d in dmas if d['instr_type']=='MTE2']
        if op == 'sigmoid_fp32':
            if not reads or reads[0][1] != 64:
                continue
            reads = reads[1:]  # Sigmoid tiling metadata precedes its first input read.
        if not reads:
            continue
        read,size = reads[0]
        amount = size//(2 if op==OPS[0] else 4)
        groups = ceil(amount/64)
        starts = [e['ts'] for e in ev if e['name']=='RV_SMOV']
        if not starts:
            continue
        stop = starts[1] if len(starts)>1 else float('inf')
        tile = [e for e in ev if e['name'].startswith('RV_') and e['ts']<stop
                and not json.loads(e['args']['detail']).get('is_duration_excluded')]
        stores = [e for e in tile if e['name']=='RV_VST']
        if len(stores)!=groups:
            continue
        if op==OPS[0]:
            # Original calibration used the union of valid SIMD events.
            end = busy = 0
            for e in tile:
                finish = e['ts']+e['dur']
                busy += max(0,finish-max(e['ts'],end))
                end = max(end,finish)
            observed = busy*1650
        else:
            observed = (max(e['ts']+e['dur'] for e in stores) - read['ts']-read['dur'])*1650
        stages.append(dict(op=op,chip=chip,shape=[int(m),int(n)],terminal='core0.veccore0',elements=amount,
                           ca_cycles=observed,trace_sha256=hashlib.sha256(data).hexdigest()))
    if not stages:
        raise RuntimeError('Archived vector21 raw traces required for --extract.')
    for name in ('development','independent'):
        bundle=json.load(gzip.open(ROOT/f'calibration/token_quant/{name}.json.gz'))
        for o in bundle['observations']:
            c=o['receipt']['case']
            p=o['terminal_phases']
            read,write=p['dma'][1:3]
            assert read['kind']=='MTE2' and write['kind']=='MTE3' and read['bytes']==2*write['bytes']
            stages.append(dict(op=OPS[2],chip=c['chip'],shape=c['shape'],terminal=p['core'],
                               elements=write['bytes'],ca_cycles=write['issue']-read['retire'],
                               receipt_sha256=o['measurement']['receipt_sha256']))
    (OUT/'nonfa-stage-evidence.json').write_text(json.dumps(stages,indent=2)+'\n')


def evaluate():
    rows=[]
    for i,line in enumerate((ROOT/'calibration/reference/windows.jsonl').read_text().splitlines(),1):
        row=json.loads(line)
        op=row['op']
        if op not in OPS:
            continue
        costs=[estimate(op,row['shape'],attrs,chip=row['chip']).latency_us for attrs in
               ({},{'vector_model':'instruction_classes'})]
        assert abs(costs[0]-row['predicted_us'])<1e-7
        rows.append(dict(op=op,chip=row['chip'],shape=row['shape'],measure='whole_us',ca=row['ca_us'],
                         original=costs[0],classes=costs[1],reference_line=i))
    for record in json.loads((OUT/'nonfa-stage-evidence.json').read_text()):
        op=record['op']; chip=load_chip(record['chip']); n=record['elements']
        g=ceil(n/64)
        scale=27/(64*1.65)/(chip.vector_tops['32']/(chip.vector_cores*chip.frequency_ghz))
        if op==OPS[0]:
            orig=g*113/64*scale
            new=tile_service('cast',n,chip)[0]
        elif op==OPS[1]:
            orig=(79+4*(g-1))*scale
            new=74+tile_service('sigmoid',n,chip)[0]
        else:
            params=coefficients()
            orig=params['vector_setup_cycles']+params['vector_cycles_per_element']*n*scale
            new=params['vector_setup_cycles']+tile_service('token_quant',n,chip,rows=n//record['shape'][-1])[0]
        rows.append(dict(op=op,chip=chip.name,shape=record['shape'],measure='matched_tile_cycles',
                         ca=record['ca_cycles'],original=orig,classes=new,reference_line=None))
    with (OUT/'nonfa-vector-comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    summary=[]
    for op,chip,unit in sorted({(r['op'],r['chip'],r['measure']) for r in rows}):
        cohort=[r for r in rows if (r['op'],r['chip'],r['measure'])==(op,chip,unit)]
        def metrics(key):
            return [100*sum(abs(r[key]-r['ca']) for r in cohort)/sum(r['ca'] for r in cohort),
                    max(100*abs(r[key]/r['ca']-1) for r in cohort)]
        summary.append(dict(op=op,chip=chip,measure=unit,n=len(cohort),baseline_wape_max=metrics('original'),
                            classes_wape_max=metrics('classes')))
    (OUT/'nonfa-vector-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--extract',action='store_true',help='Re-extract matched tiles from local raw archives')
    if cli.parse_args().extract:
        extract()
    evaluate()
