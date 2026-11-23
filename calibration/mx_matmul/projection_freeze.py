"""Bind core counts and phase budgets to an existing same-package timing freeze."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import operator_model as api
from calibration.__main__ import FUTURE, digest, encoded, predict, read, require, seal, unseal


def extract(result):
    plan=result.projection['target_plan']
    swat=plan.get('swat',{})
    return dict(l1_k=plan.get('l1_page_k',plan['tile'][2]),l0_k=plan['tile'][2],
                l1_slots=swat.get('l1_slots'),us=result.latency_us, status=result.status, aic=result.active_aic,
                launched_aiv=result.active_aiv, producer_aiv=plan['weight_producer_aiv'],
                tile=plan['tile'], waves=plan['busiest_core_tiles'], pages=plan['pages_per_tile'],
                stages_us=result.projection['stages_us'])


def freeze_plans(frozen):
    unseal(frozen,'freeze')
    root=Path(api.__file__).resolve().parent
    files={str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(root.rglob('*'))
           if p.is_file() and p.suffix in ('.py','.json')}
    require(digest(encoded(files))==frozen['package_sha256'], 'Package differs from timing freeze')
    plans={}
    for case in frozen['cases']:
        current=extract(predict(api,case,case['chip']))
        future=extract(predict(api,case,FUTURE))
        require(current['us']==case['predicted_us'] and future['us']==case['extrapolated_970dt_us']
                and future['status']=='extrapolated','Time or extrapolation differs from timing freeze')
        plans[case['id']]={'target':current,'970dt':future}
    return seal(dict(schema=1,kind='mx-plan-freeze',freeze_sha256=frozen['sha256'],
                     created_utc=datetime.now(timezone.utc).isoformat(),plans=plans))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('freeze',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    result=freeze_plans(read(args.freeze))
    with args.output.open('x') as dest:
        json.dump(result,dest,indent=2)
        dest.write('\n')
