"""Recover block-to-core placement, activation and statistic stores from complete RMSNorm runs."""
from collections import defaultdict
from math import ceil
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read, write
OUT = ROOT/'results/rms-dispatch-audit'


def extract(row):
    source = ROOT/row['db_source']
    tokens,width = row['shape']
    db = sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    cores = {}
    with db:
        for core,sub,begin,end in db.execute(
                'SELECT ExecInstrCoreId,CoreTypeId,MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) '
                'FROM ExecutedInstructions GROUP BY 1,2'):
            cores[core,sub] = dict(core=core,lane=sub-2,begin=begin,end=end,stores=[])
        for core,sub,begin,end,data in db.execute(
                "SELECT ExecInstrCoreId,CoreTypeId,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams "
                "FROM ExecutedInstructions WHERE InstrTypeId=5 AND ExecInstrName='MOV_SRC_TO_DST_ALIGNv2' "
                "ORDER BY ExecInstrCoreId,CoreTypeId,ExecInstrTickStart"):
            info=json.loads(data)
            cores[core,sub]['stores'].append((begin,end,info['dst_addr'],info['processed_bytes']))
    db.close()
    start = min(a['begin'] for a in cores.values())
    finish = max(a['end'] for a in cores.values())
    assert abs((finish-start)/1650-row['reference_us'])<1e-5,(source,finish,start)
    # On this branch each UB tile writes rstd first (FP32, one element per row),
    # then normalised y (BF16, width elements per row). Verify bytes for every pair.
    for a in cores.values():
        writes=a.pop('stores')
        assert len(writes)%2==0 and writes
        stats=writes[::2]
        assert all(b[3]*width==2*y[3] for b,y in zip(stats,writes[1::2]))
        a['begin']-=start
        a['end']-=start
        a['stats']=[dict(issue=x-start,done=y-start,addr=addr,size=n,
                         y_issue=payload[0]-start,y_done=payload[1]-start,y_bytes=payload[3])
                    for (x,y,addr,n),payload in zip(stats,writes[1::2])]
    blocks=list(cores.values())
    base=min(a['stats'][0]['addr'] for a in blocks)
    busiest=ceil(tokens/64)
    for a in blocks:
        addr=a['stats'][0]['addr']
        assert (addr-base)%(4*busiest)==0
        a['block']=(addr-base)//(4*busiest)
        for event in a['stats']:
            event['offset']=event.pop('addr')-base
    blocks.sort(key=lambda a:a['block'])
    assert [a['block'] for a in blocks]==list(range(len(blocks)))
    assert sum(e['size'] for a in blocks for e in a['stats'])==4*tokens
    return dict(shape=row['shape'],window=finish-start,role=row['role'],prior=row['prior_aiv'],terminal=row['core'],
        terminal_lane=row['core_type']-2,launch=row['launch'],source=row['db_source'],costs=row['phase_us'],
        base_mod128=base%128,blocks=blocks)


if __name__=='__main__':
    rows=read(ROOT/'results/v0.12/evaluation.json')['rows']
    data=[];unavailable=[]
    for row in rows:
        if row['op']!='rmsnorm':
            continue
        if not (ROOT/row['db_source']).is_file():
            unavailable.append(row['db_source'])
            continue
        data.append(extract(row))
    write(OUT/'core-windows.json',dict(chip='Ascend950DT_9582',cycles_per_us=1650,rows=data,unavailable=unavailable))
    print(len(data),len(unavailable),sum(len(x['blocks']) for x in data))
