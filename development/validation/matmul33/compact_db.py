"""Verify native SQLite timing records against the independent numeric harness log."""
import os
import re
import sqlite3
from time import perf_counter
from watch import ROOT, OUT, digest, read, save

def audit(path):
    start = perf_counter()
    folder = next(p for p in path.parents if p.name.startswith("matmul_"))
    log = (folder / "run.log").read_text(errors="replace")
    old = read(folder / "result.json")
    with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as db:
        assert db.execute("PRAGMA quick_check").fetchone() == ("ok",)
        meta = dict(db.execute("SELECT * FROM KernelInfo"))
        stats = db.execute("""SELECT CoreTypeId, ExecInstrCoreId, COUNT(*), MIN(ExecInstrTickStart),
                         MAX(ExecInstrTickEnd),SUM(ExecInstrName='END'),SUM(ExecInstrName='MMAD')
                         FROM ExecutedInstructions GROUP BY CoreTypeId, ExecInstrCoreId""").fetchall()
    m,n,k = map(int, re.search(r"case=matmul M=(\d+) N=(\d+) K=(\d+) dtype=BF16", log).groups())
    count = m*n
    if not stats:
        raise ValueError(f"0 executed instructions; provider status {old['status']}, exit {old['returncode']}")
    scope = max(r[4] for r in stats) - min(r[3] for r in stats)
    us = scope / float(meta["frequency_ghz"]) / 1000
    assert len(stats) > 0 and all(r[5] == 1 for r in stats)
    assert sum(r[2] for r in stats) == int(meta["instrs_exec_num"])
    assert abs(us-float(meta["exec_time_ns"])/1000) < 1e-8 and scope == int(meta['exec_time_ticks'])
    assert f"correctness elements={count} bad=0 " in log and old["returncode"] == 0 and old["status"] == "ok"
    assert old['shape'] == [m,n,k] and abs(us-old["reference_us"]) < 1e-8
    sku = re.search(r"runtime_soc=(\S+)",log)[1]
    recorded = old["kernels"]
    assert len(recorded)==1 and recorded[0]["db_sha256"]==digest(path)
    return dict(shape=[m,n,k],sku=sku,source=str(path.relative_to(ROOT)),source_sha256=digest(path),
        native_us=us,active_physical_engines=len(stats),mmad=sum(r[6] for r in stats),kernel=meta["name"],
        object_path=meta["object_path"],archive_returncode=old["returncode"],exact_golden_elements=count,
        wall_s=old["wall_s"],sqlite_bytes=path.stat().st_size,query_and_check_s=perf_counter()-start)

def find_profiled_shapes(shapes):
    # Limit to runner manifests, not multi-GB event streams; verify shape, SKU and library symbol together.
    matches=[]
    for dirname,sub,files in os.walk(ROOT/'runs'):
        sub[:]=[name for name in sub if name not in {'profile','log','log_ca','dump','tmp','simulator','record'}
                and not name.startswith(('OPPROF_','npusim_'))]
        for name in files:
            if name not in ('run.log','matmul.log') and not (name.startswith('matmul_') and name.endswith('.log')):continue
            log=ROOT/dirname/name
            if log.stat().st_size>6_000_000:continue
            raw=log.read_text(errors='replace')
            geom=re.search(r'case=matmul (?:api=aclnnMatmul )?M=(\d+) N=(\d+) K=(\d+)',raw)
            provider=re.search(r'Start profiling on kernel: (\S+)',raw)
            sku=re.search(r'runtime_soc=(\S+)',raw)
            if not (geom and provider and sku):continue
            shape=tuple(map(int,geom.groups()))
            if shape in shapes:
                matches.append(dict(shape=list(shape),sku=sku[1],kernel=provider[1],source=str(log.relative_to(ROOT))))
    return sorted(matches,key=lambda r:(r['shape'],r['source']))

def main():
    paths = [p for name in ('v8cube','v8cube-isolated') for p in (ROOT/'runs'/name).glob('matmul_*/profile/*/record/*.db')]
    rows=[];failed=[]
    for p in sorted(paths):
        try:
            rows.append(audit(p))
        except (AssertionError, ValueError, KeyError, IndexError, OSError, TypeError) as e:
            failed.append(dict(file=str(p.relative_to(ROOT)),error=repr(e)))
    save("compact-db.json",dict(scope="Installed 950DT v8cube and v8cube-isolated native Matmul record directories. Revalidate SQLite closure, native stage and all END, gold and original hash independently.",rows=rows,failed=failed,exact_geometry_msopprof_matches=find_profiled_shapes({tuple(r["shape"]) for r in rows})))
if __name__=="__main__": main()
