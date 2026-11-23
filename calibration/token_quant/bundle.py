"""Archive the frozen row-FP8 forecast, numerical receipts and measured phase boundaries."""
import argparse
import gzip
from math import isclose
from pathlib import Path
from time import perf_counter_ns

import operator_model
from calibration.__main__ import FUTURE, encoded, ingest, predict, read, require, unseal
from calibration.mx_matmul.collect import write_json


def pack(root, dev):
    frozen=unseal(read(root/('replay-1.6.json' if dev else 'freeze-1.6.json')),'freeze')
    raw=root/('dev-qualified' if dev else 'independent')
    observed=[]
    for case in frozen['cases']:
        folder=raw/case['id']
        receipt=read(folder/'receipt.json')
        measurement=ingest(frozen,receipt)
        require(measurement['data_role']==('replay' if dev else 'prospective'),'Collection role changed')
        if not dev:
            require(measurement==read(folder/'measurement.json'),'Measurement mismatch')
        cost=read(folder/'cost.json')
        if not (folder/'query-970.json').exists():
            chip=operator_model.load_chip(FUTURE)
            start=perf_counter_ns()
            result=predict(operator_model,case,chip)
            elapsed=(perf_counter_ns()-start)/1e6
            require(isclose(result.latency_us,case['extrapolated_970dt_us'],rel_tol=1e-12),'Future price changed')
            write_json(folder/'query-970.json',{'query_970_ms':elapsed})
        observed.append(dict(receipt=receipt,measurement=measurement,wall_seconds=cost['wall_seconds'],
                             query_970_ms=read(folder/'query-970.json')['query_970_ms'],probe_sha256=cost['probe_sha256'],
                             terminal_phases=read(folder/'phases.json')))
    return dict(freeze=frozen,observations=observed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--development',action='store_true')
    a=p.parse_args()
    a.output.write_bytes(gzip.compress(encoded(pack(a.raw,a.development)),mtime=0))
