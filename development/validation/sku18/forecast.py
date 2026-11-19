"""Read cases from stdin. In old-wheel mode run with python -I, outside the source tree."""
import json
from pathlib import Path
import sys
from time import perf_counter_ns


def main():
    which = sys.argv[1]
    if which == 'source':sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    before = perf_counter_ns()
    from operator_model import estimate, __version__
    import_ms = (perf_counter_ns() - before) / 1e6
    if which == 'prior' and __version__ != '0.17.0':raise RuntimeError('v0.17 wheel required')
    jobs = json.load(sys.stdin)
    output = []
    for j in jobs:
        result = {'op':j['op'], 'id':j['id'], 'shape':j['shape'], 'targets':{}}
        mode = 'auto' if which=='prior' and j['op']=='rmsnorm' else 'resource'
        for alias, chip in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b'),('future','Ascend970DT')]:
            start = perf_counter_ns()
            r = estimate(j['op'],j['shape'],j['attrs'],dtype=j['dtype'],chip=chip,timing_mode=mode)
            elapsed = (perf_counter_ns()-start)/1e6
            if not r.latency_us:raise RuntimeError((j['id'],chip,r.status,r.reason))
            result['targets'][alias] = dict(mode=mode,chip=chip,latency_us=r.latency_us,
                active_aic=r.active_aic,active_aiv=r.active_aiv,status=r.status,query_ms=elapsed,
                work=r.work,projection=r.projection)
            if which=='source':
                automatic=estimate(j['op'],j['shape'],j['attrs'],dtype=j['dtype'],chip=chip)
                result['targets'][alias]['auto_us']=automatic.latency_us
        output.append(result)
    print(json.dumps(dict(version=__version__,import_ms=import_ms,route=which,cases=output),allow_nan=False))

if __name__=='__main__':main()
