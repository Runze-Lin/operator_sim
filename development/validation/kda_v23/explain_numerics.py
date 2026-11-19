"""Replay the same real inputs on CPU and compare predicted old failures with original logs."""
import argparse
import json
from pathlib import Path
from reference import prepare,diff

BASE=Path(__file__).resolve().parents[3]/'results/kda-v23'


def main():
    rows=[]
    for source in sorted((BASE/'numeric-detail').glob('*.json')):
        old=json.loads(source.read_bytes());h,t=old['shape'][3],old['shape'][1]
        pub,math=prepare(t,h)
        groups={}
        for name,x,y in zip(('aqk','akk'),pub,math):
            e=diff(x,y);observed=old['groups'][name];maximum=e['max_abs']
            groups[name]=dict(published_vs_math=e,raw_CA_old_bad=observed['bad'],raw_CA_old_max_abs=observed['max_abs_error'],
              same_bad_count=e['bad']==observed['bad'],same_max_abs_within_log_rounding=abs(maximum-observed['max_abs_error'])<=max(1e-12,abs(maximum)*5.1e-6))
        rows.append(dict(case=old['id'],source=source.relative_to(BASE).as_posix(),group=groups))
    target=BASE/'numeric-explanation.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(role='two aggregated summaries per auxiliary tensor; diagnostic corroboration only; no change to the original sample eligibility',rows=rows),indent=2)+'\n')
    tmp.replace(target)

if __name__=='__main__':main()
