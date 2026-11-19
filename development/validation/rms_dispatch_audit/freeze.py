"""Seal alternate RMS forecasts before the original independent CA queue starts."""
from pathlib import Path
from time import perf_counter_ns
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,digest,now,read,write
from development.validation.rms_dispatch_audit.extract import OUT
from development.validation.rms_dispatch_audit.schedule import both
from operator_model import load_chip,estimate

SEQUENCE=[(1,7040),(2,6688),(3,5440),(4,4800),(16,1776),
          (34,1904),(64,1984),(65,1808),(135,2992),(504,5056)]


def register():
    primary=ROOT/'results/v0.21/rms/freeze.json'
    path=OUT/'freeze.json'
    if path.is_file() or (ROOT/'runs/rms21').exists() or (ROOT/'runs/rms-ordered-audit').exists():
        raise FileExistsError('Capture already started or registration exists')
    source=read(primary)
    known={tuple(r['shape']) for r in read(ROOT/'results/v0.21/evaluation.json')['rows'] if r['op']=='rmsnorm'}
    known.update(tuple(j['shape']) for j in source['jobs'])
    known.update(tuple(s) for s in read(ROOT/'results/v0.18/sku/excluded-history.json')['shapes']['rmsnorm'])
    if known.intersection(SEQUENCE):
        raise ValueError('Sequence already observed: '+str(known.intersection(SEQUENCE)))
    profiles=read(OUT/'candidate.json')['parameters']
    records=[]
    for name,shapes in [('isolated frozen v0.21',[j['shape'] for j in source['jobs']]),('ordered new sequence',SEQUENCE)]:
        for side,chipname in CHIPS.items():
            chip=load_chip(chipname);high=0
            for i,shape in enumerate(shapes):
                start=perf_counter_ns()
                r=estimate('rmsnorm',shape,chip=chip)
                ks,unknown=both(shape,chip,high,profiles,r)
                wall_ms=(perf_counter_ns()-start)/1e6
                assert wall_ms<100
                prev=estimate('rmsnorm',shape,{'resident_aiv':high},chip=chip)
                records.append(dict(op='rmsnorm',chip=chip.name,shape=list(shape),protocol=name,i=i,prior_aiv=high,
                        active_aiv=r.active_aiv,previous_known_us=prev.latency_us,previous_unknown_us=r.latency_us,
                        candidate_known_us=ks,candidate_unknown_us=unknown,query_ms=wall_ms))
                if name=='ordered new sequence':
                    high=max(high,r.active_aiv)
    src=[Path(__file__),Path(__file__).with_name('schedule.py'),Path(__file__).with_name('extract.py'),ROOT/'build/vector-v8/NORM',
         ROOT/'smoke/vector_v8.cpp',ROOT/'smoke/extension_norm_addmm.cpp',primary,OUT/'candidate.json',
         OUT/'core-windows.json',*(ROOT/'configs/chips').glob('*.json')]
    write(path,dict(utc=now(),rows=records,files={str(p.relative_to(ROOT)):digest(p) for p in src},
            params=profiles,measured_skus=[CHIPS['dt'],CHIPS['pr']],future_label='970DT extrapolation only',
            isolation='All isolated targets were already frozen for v0.21, new alternate outputs before either CA endpoint started. Ordered cases are separately registered and disjoint.',
            wall_per_batch_s=3600,raw_mib_per_batch=3072,concurrency=1))


if __name__=='__main__':
    register()
