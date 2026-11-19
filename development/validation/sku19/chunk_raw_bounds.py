"""Recover launch boundaries from official popped/retired dumps and verify against full export."""
from hashlib import sha256
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write,digest
from development.validation.sku19.prefill_retry import TARGET,FOLDER
FILE=TARGET/'chunk-retry-raw-bounds.json'
VALIDATION=TARGET/'chunk-retry-bound-comparison.json'
PATTERN=re.compile(rb'^\[info\] \[(\d+)\] \(PC: (0x[0-9a-f]+)\).*?\(ID: .*?\) (\w+)')

def capture():
    if FILE.exists():raise ValueError('Raw boundary record exists')
    tick_rows={};size=0;t0=perf_counter()
    retire=list(FOLDER.rglob('*.instr_log.dump'))
    for end_path in retire:
        first_path=end_path.with_name(end_path.name.replace('.instr_log.dump','.instr_popped_log.dump'))
        with first_path.open('rb') as f:first=f.readline(2048)
        with end_path.open('rb') as f:
            f.seek(-min(end_path.stat().st_size,4096),2)
            tail=f.read(4096)
        size+=len(tail)+len(first)
        end=tail.splitlines()[-1]
        a=PATTERN.match(first);b=PATTERN.match(end)
        if not (a and b and a[3]==b'MOV_XD_SPR' and b[3]==b'END_LABEL'):
            raise ValueError('Missing endpoint on '+end_path.name)
        c=end_path.name.removesuffix('.instr_log.dump')
        if c in tick_rows:raise ValueError('duplicate core')
        tick_rows[c]=dict(first_popped_tick=int(a[1]),last_retired_tick=int(b[1]),
                          first_pc=a[2].decode(),last_pc=b[2].decode(),
                          popped_first_line_sha256=sha256(first.rstrip(b'\n')).hexdigest(),
                          retired_last_line_sha256=sha256(end).hexdigest())
    if len(tick_rows)!=56:raise ValueError('expected 56 cores')
    result=dict(source='Vendor per-core *.instr_popped_log.dump first line and *.instr_log.dump last line',
                shape=[2048,5120],chip='Ascend950PR_957b',rows=tick_rows,
                endpoint_bytes_read=size,wall_ms=(perf_counter()-t0)*1000,
                theoretical_us_per_tick=1/1650)
    write(FILE,result)

def check():
    raw=read(FILE);record=read(FOLDER/'qualified.json');stage=record['stages'][0]
    w=stage['windows'];rows=raw['rows']
    if set(rows)!=set(w):raise ValueError('mismatched active cores')
    t0=min(r['first_popped_tick'] for r in rows.values()); t1=max(r['last_retired_tick'] for r in rows.values())
    err={}
    for c,r in rows.items():
        err[c]=[(r['first_popped_tick']-t0)/1650-w[c][0],(r['last_retired_tick']-t0)/1650-w[c][1]]
    maximum=max(abs(v) for pair in err.values() for v in pair)
    write(VALIDATION,dict(eligible_core_set=len(w),raw_bytes=raw['endpoint_bytes_read'],
        first_tick=t0,last_tick=t1,raw_ca_us=(t1-t0)/1650,official_ca_us=stage['ca_us'],
        max_endpoint_difference_us=maximum,within_one_cycle=maximum<=1/1650,
        per_core_error_us=err,full_trace_sha256=stage['trace_sha256'],source_sha256=digest(FILE)))

if __name__=='__main__':
    {'capture':capture,'check':check}[sys.argv[1]]()
