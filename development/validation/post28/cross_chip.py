"""Compare two measured SKUs at the same position of the same actual BF16 stream."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.28/post'
DT, PR = 'Ascend950DT_9582', 'Ascend950PR_957b'


def paired(records):
    by_call = defaultdict(dict)
    for row in records:
        trace = Path(row['original_window'])
        folder = ROOT.joinpath(*trace.parts[:3])
        sequence = (folder / 'batch.txt').read_bytes()
        shapes = []
        for i,line in enumerate(sequence.decode().splitlines()):
            op, ordinal, *dims = line.split()
            assert op == 'mhc_post' and int(ordinal) == i
            shapes.append(list(map(int, dims)))
        matches = [i for i,s in enumerate(shapes) if s == row['shape'] and
                   (shapes[i-1] if i else None) == row['previous_shape']]
        if len(matches) != 1:
            raise ValueError(('ambiguous pair',folder,matches))
        key = (row['cohort'], trace.parts[1], folder.name.rsplit('-Ascend',1)[0],
               hashlib.sha256(sequence).hexdigest(), matches[0])
        if row['chip'] in by_call[key]:
            raise ValueError(('duplicate pair',key))
        by_call[key][row['chip']] = row
    return by_call


def summary(pairs):
    observed = [r['ratio_dt_us_over_pr_us']['ca_us'] for r in pairs]
    clear = [i for i,v in enumerate(observed) if v < 1/1.05 or v > 1.05]
    result = {}
    for source in ('selected_us','old_us','stateless_us'):
        estimated = [r['ratio_dt_us_over_pr_us'][source] for r in pairs]
        result[source] = {
            'wape_pct': 100*sum(abs(a-b) for a,b in zip(estimated,observed))/sum(observed),
            'max_ape_pct': max(100*abs(a/b-1) for a,b in zip(estimated,observed)),
            'direction_correct_at_5pct': sum((estimated[i]-1)*(observed[i]-1)>0 for i in clear),
        }
    return dict(paired_windows=len(pairs), measured_ratios_outside_5pct=len(clear), models=result)


def main():
    source = OUT/'history-replay.json'
    records = json.loads(source.read_text())['records']
    groups = defaultdict(list)
    singles = []
    for key,two in paired(records).items():
        cohort, archive, sequence_name, sequence_sha, index = key
        if len(two) != 2:
            singles.append(dict(cohort=cohort, archive=archive, group=sequence_name, launch_index=index,
                                sku=next(iter(two)),batch_sha256=sequence_sha))
            continue
        a,b = two[DT],two[PR]
        assert a['shape']==b['shape'] and a['previous_shape']==b['previous_shape']
        groups[cohort].append(dict(
            cohort=cohort, archive=archive, group=sequence_name, launch_index=index, batch_sha256=sequence_sha,
            shape=a['shape'], previous_shape=a['previous_shape'],
            dt={k:a[k] for k in ('scenario','kernel','ca_us','selected_us','old_us','stateless_us','sha256','original_window')},
            pr={k:b[k] for k in ('scenario','kernel','ca_us','selected_us','old_us','stateless_us','sha256','original_window')},
            ratio_dt_us_over_pr_us={k:a[k]/b[k] for k in ('ca_us','selected_us','old_us','stateless_us')}))
    result = dict(
        scope='Original complete 950DT and 950PR device windows on an identical full ordered input batch (byte-for-byte SHA), '
              'same launch ordinal, shape and predecessor. Ratio is DT time / PR time. No inferred windows.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        metrics={key:summary(rows) for key,rows in groups.items()},
        rows=[row for rows in groups.values() for row in rows], unmatched=singles)
    (OUT/'cross-chip.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    main()
