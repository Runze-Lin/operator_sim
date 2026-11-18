"""Read only the last-finishing core; keep disjoint exposed queue time and counts."""
from collections import Counter, defaultdict
import json, sqlite3, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/v0.10/wide'

def inspect(path):
    path=Path(path)
    if path.suffix=='.json':path=Path(json.loads(path.read_text())['measurement']['kernels'][0]['db_path'])
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        info=dict(db.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
        origin=db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()[0]
        cid,typ,begin,end=db.execute('SELECT ExecInstrCoreId,CoreTypeId,MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions GROUP BY 1,2 ORDER BY 4 DESC,1,2 LIMIT 1').fetchone()
        events=defaultdict(Counter);counts=Counter();copies=Counter();dur=Counter();slow=[]
        for q,a,z,name,addr,skip in db.execute("SELECT json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName,SourceInstrAddr,ExecInstrIsDurationExcluded FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=?",(cid,typ)):
            counts[q]+=1
            if skip:continue
            if q=='MTE2':copies[name]+=1
            slow.append((z-a,name));dur[q]+=z-a
            events[a][q]+=1;events[z][q]-=1
    events[begin];events[end];last=begin;busy=Counter();out=Counter();
    for tick,event in sorted(events.items()):
        q=next((q for q in ('SIMD','MTE3','MTE2','SCALAR') if busy[q]>0),'other');out[q]+=(tick-last)/1650; busy.update(event);last=tick
    out['arrival']=(begin-origin)/1650
    assert abs(sum(out.values())-(end-origin)/1650)<1e-8
    return dict(phase_us=out,counts=counts,copies=copies,raw_duration_ticks=dur,slow=sorted(slow,reverse=True)[:8],launch=int(info['launch_id']),core=cid,core_type=typ,db_source=str(path.relative_to(ROOT) if path.is_absolute() else path))

def main():
    rows=[]
    for row in json.loads((ROOT/'results/v0.9/vector/rows.json').read_text()):
        if row['op'] not in ('rmsnorm','mhc_post'):continue
        src=next((e['kernels'][0]['db_path'] for e in json.loads((ROOT/'results/v0.7/evaluation.json').read_text())['rows'] if e['id']==row['id']),row['source']);r={**row,**inspect(ROOT/src)};rows.append(r)
        print(r['id'],r['shape'],r['active_aiv'],r['launch'],*[round(r['phase_us'].get(q,0),2) for q in ('SCALAR','MTE2','SIMD','MTE3','arrival')],tuple(r['counts'].get(q,0) for q in ('SCALAR','MTE2','SIMD')),r['core'],flush=True)
    (OUT/'development-trace.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
