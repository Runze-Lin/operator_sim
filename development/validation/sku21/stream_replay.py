"""Price archived launches with only scheduler information available before dispatch."""
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import OperatorStream, estimate
from development.validation.sku21.acceptance import CHIPS, OUT, score, read, write


def replay():
    rows = []
    launches = defaultdict(list)
    for r in read(ROOT/'results/v0.12/evaluation.json')['rows']:
        if r['op'] in {'rmsnorm', 'mhc_post'}:
            launches[r['op'], Path(r['db_source']).parent].append(r)
    omitted = []
    for (op, folder), seq in launches.items():
        ordered = sorted(seq, key=lambda r: r['launch'])
        if [r['launch'] for r in ordered] != list(range(len(seq))):
            omitted.append(str(folder))
            continue
        stream = OperatorStream(CHIPS['dt'])
        for r in ordered:
            known = stream.advance(op, r['shape'], r['attrs'])
            unknown = estimate(op, r['shape'], r['attrs'], chip=stream.chip)
            rows.append(dict(op=op, chip=CHIPS['dt'], protocol='ordered compact stream',
                shape=r['shape'], launch=r['launch'], us=known.latency_us, unknown_us=unknown.latency_us,
                ca_us=r['reference_us'], source=r['db_source'], active_aiv=known.active_aiv,
                state=known.projection['timing_diagnostics'].get('residency', known.projection['timing_diagnostics'].get('context'))))
    for r in read(ROOT/'results/v0.19/evaluation.json')['rows']:
        if r['op'] not in {'rmsnorm', 'mhc_post'}:
            continue
        for side in ('dt','pr'):
            obs = r[side+'_observation'] or {}
            paths = obs.get('trace',[])
            host = Path(paths[0]).parts if len(paths)==1 else ()
            # Each host directory embeds that one operator; multi-launch/chains are excluded.
            if 'profile' not in host or not host[host.index('profile')-1].startswith(r['op']+'_'):
                continue
            stream = OperatorStream(CHIPS[side])
            result = stream.advance(r['op'], r['shape'], r['attrs'])
            rows.append(dict(op=r['op'], chip=CHIPS[side], protocol='isolated official launch',
                shape=r['shape'], launch=0, us=result.latency_us, state='first',
                unknown_us=estimate(r['op'],r['shape'],r['attrs'],chip=CHIPS[side]).latency_us,
                ca_us=r[side+'_ca_us'],source=paths[0],active_aiv=result.active_aiv))
    by_protocol = defaultdict(list)
    for r in rows:
        by_protocol[r['op'],r['chip'],r['protocol']].append(r)
    summary = [dict(op=op,chip=chip,protocol=protocol,stream=score(group),unknown=score(group,'unknown_us'))
               for (op,chip,protocol),group in sorted(by_protocol.items())]
    write(OUT/'dispatch/stream-replay.json',dict(role='development replay; timed shape-only calls in original order, no measured active cores supplied to model',summary=summary,rows=rows,omitted_nonconsecutive_runs=omitted))
    return summary


if __name__ == '__main__':
    for r in replay():
        print(r['op'],r['chip'],r['protocol'],r['stream']['n'],round(100*r['stream']['wape'],2),round(100*r['stream']['max_ape'],2))
