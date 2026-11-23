"""Look up archived actual timings, identities and projections by operator and geometry."""
import argparse
import json
from .__main__ import HERE

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('op',help='Public operator name, e.g. matmul or mhc_post')
    cli.add_argument('--shape',type=int,nargs='+')
    cli.add_argument('--contract',help='Optional exact contract label from reference/summary.json')
    cli.add_argument('--chip')
    args=cli.parse_args()
    matches=0
    with (HERE/'reference/windows.jsonl').open() as file:
        for line in file:
            row=json.loads(line)
            if row['op']==args.op and (not args.shape or row['shape']==args.shape) and (
                not args.contract or row['contract']==args.contract) and (not args.chip or row['chip']==args.chip):
                print(line.rstrip())
                matches+=1
    from .diagnostics import rows as measured_only
    for row in measured_only():
        if row['op']==args.op and (not args.shape or row['shape']==args.shape) and (
            not args.contract or row['contract']==args.contract) and (not args.chip or row['chip']==args.chip):
            print(json.dumps(row,ensure_ascii=False))
            matches+=1
    if matches==0:cli.exit(1,'No qualified matching CA window.\n')
