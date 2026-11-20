"""Observe stages at original device clocks, without short-window or core-count proxies."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.34/indexer'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def summarize(events, lo, hi):
    lanes = defaultdict(list)
    hist = Counter()
    sort_lanes = defaultdict(list)
    sort_markers = Counter()
    counts = Counter()
    for core, name, start, finish, detail in events:
        if name == 'END':
            counts[core] += 1
        elif name == 'MMAD':
            lanes[core].append([start-lo,finish-lo]);hist[(detail['m'],detail['n'],detail['k'])] += 1
        elif name == 'RV_CHISTv2':
            if finish > start:
                sort_lanes[core].append([start-lo,finish-lo])
            else:
                sort_markers[core] += 1
    assert counts and set(counts.values()) == {1}
    assert lanes and all(core in counts for core in lanes)
    for cells in lanes.values():
        cells.sort()
    last = max(lanes,key=lambda c:lanes[c][-1][1]);cells=lanes[last]
    last_sort = max(sort_lanes, key=lambda c:max(e[1] for e in sort_lanes[c]))
    return dict(launched=len(counts),useful_aic=len(lanes),histogram_aiv=len(sort_lanes),
                total_histogram_instructions=sum(map(len,sort_lanes.values())),
                zero_duration_histogram_markers=sum(sort_markers.values()),
                mmads={str(c):len(v) for c,v in lanes.items()},instruction_shapes=[[*key,n] for key,n in sorted(hist.items())],
                last_cube=str(last),critical_mmad_ticks=cells,earliest_mmad=min(v[0][0] for v in lanes.values()),
                last_sort_lane=str(last_sort), sort_ticks=sorted(sort_lanes[last_sort]),
                last_mmad_retire=cells[-1][1],duration_ticks=hi-lo,
                after_mmad_ticks=hi-lo-cells[-1][1])


def db_rows():
    for row in json.loads((ROOT/'results/v0.8/fusion/rows.json').read_bytes()):
        if row['op']!='lightning_indexer':continue
        receipt=json.loads((ROOT/row['source']).read_bytes()); k,=receipt['measurement']['kernels']; path=ROOT/k['db_path']
        h=digest(path);assert row['db_sha256']==[h] and k['db_sha256']==h
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            lo, hi=db.execute('SELECT MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
            assert abs((hi-lo)/1650-row['reference_us'])<1e-7
            records=db.execute("SELECT CoreTypeId,ExecInstrCoreId,ExecInstrName,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams FROM ExecutedInstructions WHERE ExecInstrName IN ('MMAD','RV_CHISTv2','END')")
            result=summarize(((f'{kind}-{core}',name,a,b,json.loads(detail)) for kind,core,name,a,b,detail in records),lo,hi)
        yield dict(chip='Ascend950DT_9582',sku_role='archive',shape=row['shape'],sha=h,source=str(path.relative_to(ROOT)),protocol='npusim',kernel=k['kernel'],**result)


def device_rows():
    for path in sorted((ROOT/'runs/sku17/extension').glob('lightning_indexer*/strict.json')):
        row,=json.loads(path.read_bytes()); name,=row['trace'];trace=ROOT/name;h=digest(trace);assert [h]==row['trace_sha256']
        rows=json.loads(trace.read_bytes())['traceEvents'];rows=[e for e in rows if e.get('ph')=='X']
        lo=min(round(e['ts']*1650) for e in rows);hi=max(round((e['ts']+e['dur'])*1650) for e in rows)
        assert abs((hi-lo)/1650-row['ca_us'])<2/1650
        histograms = [e for e in rows if e['name'] == 'RV_CHISTv2']
        def pcs(positive_duration):
            return Counter((e['pid'], e['args']['pc_addr']) for e in histograms
                           if (e['dur'] > 0) == positive_duration)
        assert pcs(True) == pcs(False), 'issued markers and execution records differ by lane or instruction address'
        result=summarize(((e['pid'],e['name'],round(e['ts']*1650),round((e['ts']+e['dur'])*1650),json.loads(e['args']['detail'])) for e in rows if e['name'] in ('MMAD','RV_CHISTv2','END')),lo,hi)
        yield dict(chip=row['chip'],sku_role='archive',shape=row['shape'],sha=h,source=str(trace.relative_to(ROOT)),protocol='msopprof',kernel=row['kernel'],**result)


if __name__ == '__main__':
    data=[*db_rows(),*device_rows()]; assert len(data)==len({r['sha'] for r in data})==24
    (OUT/'physical.json').write_text(json.dumps(data,indent=2)+'\n')
