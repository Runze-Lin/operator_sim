"""Check each launch independently after a mixed-success helper batch."""
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_db

OUT = ROOT / 'results/v0.11/post'


def extract(folder, cases, wall_s):
    raw, = folder.rglob('npusim.log')
    cli = (folder / 'run.log').read_text()
    body = raw.read_text()
    assert 'SMOKE_FAIL' not in cli and 'Simulation SUCCESS' in cli and 'SMOKE_PASS' in cli
    assert re.findall(r'runtime_soc=(\S+)', body) == ['Ascend950DT_9582']
    assert not re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)', body)
    helper, = folder.rglob('cannsim_helper.log')
    flushes = re.findall(r'instruction handler flush: ([^\n]*)', helper.read_text())
    dbs = sorted(folder.rglob('*.db'))
    assert len(cases) == len(dbs) == len(flushes)
    good, failed = [], []
    for i, (case, db, flush) in enumerate(zip(cases, dbs, flushes)):
        segment, = re.findall(rf'BATCH_BEGIN {i}\n(.*?)BATCH_END {i}\n', body, re.S)
        assert len(set(re.findall(r'\[TASK_DONE\].*?stream_id:(\d+), task_id:(\d+) TASK_DONE', segment))) == 1
        checks = re.findall(r'correctness [^\r\n]+', segment)
        assert len(checks) == 1 and ' bad=0 ' in checks[0]
        assert int(re.search(r'count=(\d+)', checks[0])[1]) == case['shape'][0] * case['shape'][1] * case['shape'][2]
        row = {**case, 'source':str(db.relative_to(ROOT)), 'correctness':checks, 'helper_flush':flush,
               'batch_wall_s':wall_s, 'batch':folder.name, 'db_sha256':sha256(db)}
        if 'status=failed' in flush:
            with sqlite3.connect(db.resolve().as_uri() + '?mode=ro',uri=True) as con:
                count = con.execute('SELECT COUNT(*) FROM ExecutedInstructions').fetchone()[0]
            assert count == 0 and ' written=0 ' in flush
            failed.append({**row, 'role':'timing_failed', 'reference_us':None})
            continue
        assert 'invalid=0' in flush and 'status=ok' in flush
        k = extract_db(db)
        assert int(k['helper_info']['launch_id']) == i and k['reference_us'] > 0
        good.append({**row, 'reference_us':k['reference_us'], 'predicted_us':case['prediction']['predicted_us'],
                     'active_aic':k['active_aic'], 'active_aiv':k['active_aiv'], 'kernel':k['kernel'],
                     'max_core_rv_count':k['max_core_rv_count']})
    return good, failed


def main():
    frozen = json.loads((OUT / 'freeze.json').read_text())
    for path, digest in frozen['sha256'].items():
        assert sha256(ROOT / path) == digest
    folder = ROOT / 'runs/v10wide/mhc_post-0-10'
    cli = (folder / 'run.log').read_text()
    wall = float(re.search(r'Simulation SUCCESS\s+·\s+run time ([\d.]+)s', cli)[1])
    good, failed = extract(folder, frozen['cases'], wall)
    (OUT / 'original-attempt.json').write_text(json.dumps(dict(completed=good, timing_failed=failed, wall_s=wall),
                                                          indent=2) + '\n')
    print('Original CA batch', len(good), 'qualified', len(failed), 'failed', wall)


if __name__ == '__main__':
    main()
