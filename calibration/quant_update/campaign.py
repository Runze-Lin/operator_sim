"""Run at most one frozen DT/PR quant CA at a time; retain all attempts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import monotonic
from calibration.__main__ import read, require, unseal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for opt in ('freeze','raw_root','probe','opp','launcher'):
        parser.add_argument('--'+opt.replace('_','-'),type=Path,required=True)
    parser.add_argument('--budget-minutes',type=int,default=32)
    args=parser.parse_args()
    campaign=unseal(read(args.freeze),'freeze')
    start=monotonic()
    records=[]
    root=args.raw_root.resolve()
    root.mkdir(exist_ok=False)
    for case in campaign['cases']:
        raw=root/case['id']; remaining=int((60*args.budget_minutes-(monotonic()-start))//60)
        if remaining<1:
            records.append(dict(id=case['id'],state='budget-exhausted'))
        else:
            minutes=min(remaining,5 if case['shape'][0]>=576 and case['shape'][1]>=6144 else 3)
            cmd=[sys.executable,'-m','calibration.quant_mxfp4','--freeze',str(args.freeze),
                 '--raw',str(raw),'--opp',str(args.opp),'--launcher',str(args.launcher),
                 '--probe',str(args.probe),'--minutes',str(minutes),'--case',case['id']]
            p=subprocess.run(cmd,capture_output=True,text=True)
            ok=p.returncode==0 and (raw/'measurement.json').is_file()
            records.append(dict(id=case['id'],state='complete' if ok else 'failed',messages=(p.stdout+p.stderr)[-1800:]))
            print(records[-1],flush=True)
        (root/'progress.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':
    main()
