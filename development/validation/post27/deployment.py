"""Evaluate the original repeat policy on each measured physical group and on the whole cohort."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post26.freeze import OUT as V26
from development.validation.post27.issue import OUT
from development.validation.sku19.campaign import read,write,now
from development.validation.post24.report import stats
def main():
    obj=read(V26/'score.json');regions={};all_ok=True
    for sku,n in obj['all_core_repeat_shapes'].items():
        rows=[r for r in obj['independent'] if r['chip']==sku and r['role']=='repeat' and r['eligible_repeat']]
        allgroups={'all':rows};has_proof=True
        for r in rows:
            z=r['terminal_structure']
            if not z or not z['actual_matches_arch35_loop']:has_proof=False;continue
            kind=('multi-' if z['terminal_rows']>1 else 'one-')+'row / '+('multi-' if z['channel_tiles']['count']>1 else 'one-')+'channel-fragment'
            allgroups.setdefault(kind,[]).append(r)
        metrics={}
        for label,rs in allgroups.items():
            value=stats(rs,'policy_us');metrics[label]=dict(different_shapes=len({tuple(r['shape']) for r in rs}),policy=value,old=stats(rs,'public_us'))
            if value and (value['wape']>20 or value['maximum']>40):all_ok=False
        regions[sku]=dict(eligible_shapes=n,physically_accounted=has_proof,groups=metrics)
        all_ok &= bool(has_proof and n>=10 and metrics['all']['policy'] and metrics['all']['policy']['wape']<metrics['all']['old']['wape'])
    out=dict(utc=now(),qualified_for_default=bool(all_ok),acceptance='Original frozen exact-repeat prices. Each SKU needs >=10 different independent complete shapes over its full exact-repeat cohort. Each measured physical group must satisfy WAPE<=20% and max<=40%. Multiple channel fragments qualify only for row-aggregate total latency.',regions=regions)
    write(OUT/'deployment.json',out)
    return out
if __name__=='__main__':main()
