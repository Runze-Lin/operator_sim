"""Measure uncached shape+chip dispatch and prove equality against every frozen H."""
import json,sys
from pathlib import Path
from time import perf_counter_ns
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from development.validation.rms_unknown23.candidate import forecast as original
from development.validation.rms_unknown23.fast import forecast
from development.validation.sku19.campaign import write
OUT=ROOT/'results/rms-unknown23'

def main():
    frozen=json.loads((OUT/'freeze.json').read_text())['rows'];out=[];max_abs=0
    for row in frozen:
        chip=load_chip(row['chip']);t=perf_counter_ns();new=forecast(row['shape'],chip);ms=(perf_counter_ns()-t)/1e6
        error=max([abs(new[k]-row[k]) for k in ('candidate_us','previous_us','lower_us','upper_us')]+[abs(a-b) for a,b in zip(row['conditional_us_by_H'],new['conditional_us_by_H'])]);assert len(row['conditional_us_by_H'])==len(new['conditional_us_by_H']);assert error<1e-10;max_abs=max(max_abs,error)
        out.append(dict(chip=chip.name,shape=row['shape'],process=row['process'],role=row['role'],wall_ms=ms,max_abs_us=error))
    old=json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows'];tests=sorted({(r['chip'],tuple(r['shape'])) for r in old});hist=0;hs=0
    for name,sh in tests:
        chip=load_chip(name);v0=original(sh,chip);v1=forecast(sh,chip);pairs=list(zip(v0['conditional_us_by_H'],v1['conditional_us_by_H']));err=max(abs(a-b) for a,b in pairs); assert err<1e-10 and abs(v0['candidate_us']-v1['candidate_us'])<1e-10;hist=max(hist,err);hs+=len(pairs)
    write(OUT/'fast-parity.json',dict(freeze_rows=len(out),all_H_frozen=sum(len(r['conditional_us_by_H']) for r in frozen),max_abs_frozen_us=max_abs,history_cases=len(tests),all_H_history=hs,max_abs_history_us=hist,rows=out,max_wall_ms=max(r['wall_ms'] for r in out),timing_scope='stock host single call, resident Python/chip parameters, fresh local sector cache, no persistent shape cache'))
    print(len(out),max_abs,hist,max(r['wall_ms'] for r in out))
if __name__=='__main__':main()
