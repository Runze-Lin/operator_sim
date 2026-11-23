"""Build a portable freeze/observations bundle from complete native CA outputs."""
import argparse
import gzip
from math import isclose
from pathlib import Path
from time import perf_counter_ns

import operator_model
from calibration.__main__ import FUTURE, encoded, ingest, predict, read, require, unseal
from calibration.cache_load.collect import collect
from calibration.mx_matmul.collect import write_json


def bundle(freeze_file, basedir, opp):
    frozen=unseal(read(freeze_file),'freeze')
    observations=[]
    target=operator_model.load_chip(FUTURE)
    for case in frozen['cases']:
        raw=basedir/case['id']
        if not (raw/'measurement.json').exists():
            collect(case,raw,opp,frozen)
        receipt=read(raw/'receipt.json')
        measurement=read(raw/'measurement.json')
        require(measurement==ingest(frozen,receipt),'Receipt/measurement mismatch')
        cost=read(raw/'cost.json')
        query_file=raw/'query-970.json'
        if not query_file.exists():
            start=perf_counter_ns()
            us=predict(operator_model,case,target).latency_us
            ms=(perf_counter_ns()-start)/1e6
            require(isclose(us,case['extrapolated_970dt_us'],rel_tol=1e-12),'Frozen future price differs')
            write_json(query_file,{'query_970_ms':ms})
        observations.append(dict(receipt=receipt,measurement=measurement,wall_seconds=cost['wall_seconds'],
                                 probe_sha256=cost['probe_sha256'],query_970_ms=read(query_file)['query_970_ms'],
                                 terminal_phases=read(raw/'phases.json')))
    return dict(freeze=frozen,observations=observations)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','raw','opp','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    data=gzip.compress(encoded(bundle(a.freeze,a.raw,a.opp)),mtime=0)
    if a.output.exists():
        require(gzip.decompress(a.output.read_bytes())==gzip.decompress(data),'Existing archive differs')
    else:
        with a.output.open('xb') as f:
            f.write(data)


if __name__=='__main__':
    main()
