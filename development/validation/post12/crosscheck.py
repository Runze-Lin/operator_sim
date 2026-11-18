"""Hold out each historical simulator process in full; never read the sealed validation labels."""
from pathlib import Path
from freeze import OLD, OUT, ATTRS, ROOT, fit, save
from operator_model.post_v12 import estimate
from operator_model.post_v11 import cohort
from operator_model.wide_v10 import occupancy
import json
import runpy

stats=runpy.run_path(str(ROOT/'scripts/wide-v10-report.py'))['stats']

def main():
    dev=json.loads(OLD.read_text())
    for r in dev: r['process']=str(Path(r['db_source']).parents[1]); r['state']=occupancy(r['active_aiv'],r['prior_aiv'])
    rows=[]
    for process in sorted({r['process'] for r in dev}):
        p=fit([r for r in dev if r['process']!=process]); held=[r for r in dev if r['process']==process]
        for r in held:
            pred=estimate(p,'mhc_post',r['shape'],ATTRS); costs=pred['context_predictions_us']; counts=p['context_counts'][cohort(pred['route'],r['shape'][0],pred['predicted_active_aiv'])]; acc=0
            for chosen in sorted(costs,key=costs.get):
                acc+=counts[chosen]
                if 2*acc>=sum(counts.values()):break
            rows.append(dict(shape=r['shape'],process=process,reference_us=r['reference_us'],group=r['group'],state=r['state'],
                             old_rule_us=costs[chosen],new_rule_us=pred['predicted_us'],known_us=costs[r['state']]))
    groups={g:[r for r in rows if r['group']==g] for g in ('few_wide','controls')}
    groups['all']=rows
    groups.update({s:[r for r in groups['few_wide'] if r['state']==s] for s in ('first','expand','reuse')})
    metrics=lambda rr:{key:stats(rr,key) for key in ('old_rule_us','new_rule_us','known_us')}
    save('process-crosscheck.json',dict(design='10 prior processes; re-fit all phase coefficients and counts in each training fold; no new holdout labels.',rows=rows,
                                       groups={g:metrics(rr) for g,rr in groups.items()},
                                       by_process={p:metrics([r for r in rows if r['process']==p]) for p in sorted({r['process'] for r in rows})}))
    for g,rr in groups.items(): print(g,len(rr), [(k, round(v['wape'],3),round(v['max_ape'],3)) for k,v in metrics(rr).items()])

if __name__=='__main__': main()
