"""Post-hoc process-held-out diagnostic. Never writes a production profile or freezes predictions."""
from collections import Counter
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from operator_model.post_v11 import cohort, predict as frozen
from operator_model.vector_v9 import schedule
from operator_model.wide_v10 import occupancy, predict as previous

OUT = ROOT/"results/v0.11/post"
STATES = ("first","expand","reuse")
QUEUES = ("SCALAR","SIMD","MTE2","MTE3","arrival","other")
inspect = runpy.run_path(str(ROOT/"scripts/wide-v10-trace.py"))["inspect"]


def fit_route(train, route):
    """Refit the six v0.11 phase designs, pooling across states if one state is absent."""
    samples = [r for r in train if r["route"]==route]
    profile = {}
    for queue in QUEUES:
        state_terms = queue in ("arrival","SCALAR","MTE2","SIMD")
        service = queue=="SIMD" or (queue=="MTE2" and route=="post_token")
        row_term = route=="post_token" and queue in ("SCALAR","MTE3")
        labels = [s for s in STATES if any(r["state"]==s for r in samples)] if state_terms else ["all"]
        features = []
        for r in samples:
            x = [float(s in (r["state"],"all")) for s in labels]
            if service:
                x.append(r["units"][queue])
            if row_term:
                x.append(r["rows"]-1)
            features.append(x)
        mat = np.asarray(features)
        y = np.array([r["phase_us"].get(queue,0.) for r in samples])
        beta = least_squares(lambda b: mat@b-y, np.full(mat.shape[1],.01),
                             bounds=(0,np.inf),loss="soft_l1",f_scale=.1).x
        intercepts = dict(zip(labels,beta))
        profile[queue] = {s:intercepts.get(s,float(np.median(beta[:len(labels)]))) for s in STATES}
        profile[queue].update(service=beta[len(labels)] if service else 0.,rows=beta[-1] if row_term else 0.)
    return profile


def summarize(rows):
    if not rows:
        return {"n":0}
    actual = np.array([r["ref"] for r in rows])
    stats = {"n":len(rows)}
    for key in ("v10","v11","refit_hard","mixture","known"):
        error = np.array([r[key] for r in rows])-actual
        stats[key] = dict(wape=float(sum(abs(error))/sum(actual)), max_ape=float(max(abs(error)/actual)),
                          signed_bias=float(sum(error)/sum(actual)))
    return stats


def diagnostics():
    historical = [r for r in json.loads((ROOT/"results/v0.10/wide/rows.json").read_text()) if r["op"]=="mhc_post"]
    # The T=256 original DB contains preceding-launch writes. Whitelist the 7
    # audited launches; retries are separate processes and do not enter this diagnostic.
    original = [r for r in json.loads((OUT/"original-attempt.json").read_text())["completed"]
                if r["id"] in {f"v11_mhc_post_{i}" for i in range(7)}]
    assert len(original)==7 and all(r["shape"][0]<128 for r in original)
    for r in original:
        r.update(inspect(ROOT/r["source"]))
    rows = historical+original
    for r in rows:
        route, active, waves, units = schedule("mhc_post",r["shape"],r["attrs"])
        r.update(route=route,active=active,rows=waves,units=units,cohort=cohort(route,r["shape"][0],active),
                 fold=Path(r["db_source"]).parents[1].name, state=occupancy(active,r["prior_aiv"]))
    predictions=[]
    folds=sorted({r["fold"] for r in rows})
    for fold in folds:
        train = [r for r in rows if r["fold"]!=fold]
        phases = {route:fit_route(train,route) for route in ("post_channel","post_token")}
        for r in rows:
            if r["fold"]!=fold:
                continue
            peers = [t for t in train if t["cohort"]==r["cohort"]]
            priors = Counter(occupancy(r["active"],t["prior_aiv"]) for t in peers)
            global_prior = Counter(occupancy(r["active"],t["prior_aiv"]) for t in train)
            # Three effective observations from the pooled distribution limit sparse strata.
            weights = {s:(priors[s]+3*global_prior[s]/len(train))/(len(peers)+3) for s in STATES}
            totals={s:sum(coeff[s]+coeff["service"]*r["units"][queue]+coeff["rows"]*(r["rows"]-1)
                          for queue,coeff in phases[r["route"]].items()) for s in STATES}
            labels=Counter(t["state"] for t in (peers or train))
            cumulative=0
            for choice in sorted(STATES,key=totals.get):
                cumulative+=labels[choice]
                if cumulative*2>=sum(labels.values()):
                    break
            predictions.append(dict(shape=r["shape"],fold=fold,group=r["group"],state=r["state"],ref=r["reference_us"],
                v10=previous("mhc_post",r["shape"],r["attrs"])["predicted_us"],v11=frozen("mhc_post",r["shape"],r["attrs"])["predicted_us"],
                refit_hard=totals[choice],mixture=sum(weights[s]*totals[s] for s in STATES),
                known=totals[r["state"]], active=r["active"],prior_resident=r["prior_aiv"],weights=weights))
    subsets={"all":predictions,**{g:[r for r in predictions if r["group"]==g] for g in ("few_wide","controls")},
             **{s:[r for r in predictions if r["state"]==s] for s in STATES}}
    stats={k:summarize(v) for k,v in subsets.items()}
    by_fold={f:{g:summarize([r for r in predictions if r["fold"]==f and r["group"]==g]) for g in ("few_wide","controls")} for f in folds}
    assert len(predictions)==40 and len(folds)==7
    (OUT/"context-sensitivity.json").write_text(json.dumps(dict(scope="post-hoc development diagnostic",rows=predictions,
        summary=stats,folds=by_fold),indent=2)+"\n")
    print({g:[round(v[k]["wape"]*100,1) for k in ("v10","v11","refit_hard","mixture","known")] for g,v in stats.items()})
    print({f.split("_")[1]:{g:[v["n"],*[round(v[k]["wape"]*100,1) for k in ("v10","v11","refit_hard","mixture")]] if v["n"] else [] for g,v in groups.items()} for f,groups in by_fold.items()})

if __name__=="__main__":
    diagnostics()
