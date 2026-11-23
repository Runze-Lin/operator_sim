"""Replay Kimi's captured row-strided BF16 view without assuming a kernel count."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
from time import monotonic
from calibration.__main__ import seal, require
from calibration.msopprof import sha
from calibration.mx_matmul.collect import now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw','probe','launcher'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--chip', choices=('Ascend950DT_9582','Ascend950PR_957b'), default='Ascend950DT_9582')
    args = p.parse_args()
    # The descriptor includes the nonzero offset inside the 9×6288 allocation.
    # No point estimate exists for this composite route.
    raw = args.raw.resolve()
    contract = seal(dict(schema=1, kind='layout-capability', created_utc=now(), api='aclnnDynamicMxQuant',
                         chip=args.chip, shape=[9,128], stride=[6288,1], storage_offset=6144,
                         backing_bytes=113184, group_size=32, round_mode='round', scale_alg=0,
                         outputs={'Y':1152,'scale':36}, predicted_us=None, timeout_seconds=180))
    require(not raw.exists(), 'Use a fresh raw directory')
    write_json(raw.with_suffix('.contract.json'), contract)
    raw.mkdir()
    (raw/'freeze.sha256').write_text(contract['sha256']+'\n')
    cmd=['bash',str(args.launcher.resolve()),'msopprof','simulator','--launch-count=3',
         '--soc-version='+contract['chip'],'--dump=false','--timeout=3','--output='+str(raw/'profile'),
         str(args.probe.resolve()),'quant_mxfp4','9','128','6288','6144']
    started=now(); begin=monotonic()
    with (raw/'run.log').open('x') as log:
        proc=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,cwd=raw,start_new_session=True)
        try:
            status=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            status=124
    write_json(raw/'cost.json',dict(started_utc=started,finished_utc=now(),exit_code=status,command=cmd,
                                    wall_seconds=monotonic()-begin,probe_sha256=sha(args.probe)))


if __name__=='__main__':
    main()
