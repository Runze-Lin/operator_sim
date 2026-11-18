"""Read only earlier critical-core queue intervals. Do not resimulate training cases."""
from collections import defaultdict,Counter
from pathlib import Path
import sqlite3,json
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.12/rms'
def measure(r):
    with sqlite3.connect((ROOT/r['db_source']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        sql="""select ExecInstrTickStart, ExecInstrTickEnd, ExecInstrName, json_extract(ExtendParams,'$.queue_type'), json_extract(ExtendParams,'$.processed_bytes') from ExecutedInstructions where CoreTypeId=? and ExecInstrCoreId=? and ExecInstrIsDurationExcluded=0 order by ExecInstrTickStart"""
        events=defaultdict(Counter); stores=[]
        for a,b,name,q,sz in db.execute(sql,(r['core_type'],r['core'])):
            if q not in ('MTE3','SIMD'):continue
            events[a][q]+=1; events[b][q]-=1
            if q=='MTE3' and sz: stores.append(dict(bytes=sz,start=a,end=b))
    busy=simd=both=last=0;n=Counter()
    for tick,event in sorted(events.items()):
        if n['MTE3']:busy+=tick-last
        if n['SIMD']:simd+=tick-last
        if n['MTE3'] and n['SIMD']:both+=tick-last
        last=tick;n.update(event)
    assert abs((busy-both)/1650-r['phase_us']['MTE3'])<1e-7
    return dict(shape=r['shape'],source=r['db_source'],core=r['core'],core_type=r['core_type'],active=r['active_aiv'],prior=r['prior_aiv'],phases=r['phase_us'],simd_ticks=simd,write_ticks=busy,overlap_ticks=both,stores=stores)
if __name__=='__main__':
    rows=[measure(r) for r in json.loads((ROOT/'results/v0.11/rms/rows.json').read_text())]; (OUT/'development-evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(*r['shape'],r['active'],r['prior'],r['write_ticks'],r['simd_ticks'],r['overlap_ticks'],[(s['bytes'],s['end']-s['start']) for s in r['stores']])
