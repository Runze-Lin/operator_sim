"""Resume sealed Kimi acquisitions if a supervisor exits before qualification."""
import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul28 import campaign as original

# sku17 and the decode harness both have a top-level campaign.py. Bind the
# original trace reader in its own namespace before loading the worker.
reader = original.stages
assert reader.__globals__['PR'] == original.PR
assert callable(reader.__globals__['read']) and callable(reader.__globals__['sha'])
sys.modules.pop('campaign',None)
sys.path.insert(0,str(Path(__file__).parent/'decode'))
import wide_host


def main(check=False):
    base = wide_host.base
    assert base.ca.stages is reader and base.REG.parent.name == 'decode'
    if check:
        # Replay a complete, already sealed small receipt via the unmodified reader.
        folder = ROOT/'runs/matmul28/1-2304-1536-pr'
        stages,gold = reader(dict(kernels=1,nchecks=1),base.PR,folder)
        stored = base.read(base.BASE/'1-2304-1536-pr-drain.json')
        assert gold==stored['correctness'] and stages[0]['trace_sha256']==stored['stages'][0]['trace_sha256']
        assert stages[0]['ca_us']==stored['ca_us']
        return
    with (base.RAW/'standby.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The owning CA keeps its lock through every running job and its report.
        with (base.RAW/'serial.lock').open('a') as lane:
            fcntl.flock(lane,fcntl.LOCK_EX)
        if base.read(base.D/'state.json')['phase']!='finished':
            wide_host.run()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    main(parser.parse_args().check)
