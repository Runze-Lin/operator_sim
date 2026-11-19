"""Prospective deployment-policy supplement for unstarted width/many groups only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from development.validation.post25.repeat_policy import Contract,estimate_repeat
from development.validation.post24.freeze import sha
from development.validation.post25.calibrate import OUT
from development.validation.sku19.campaign import read,write,now
def main():
    out=OUT/'repeat-policy-freeze.json';reg=OUT/'freeze.json';raw=ROOT/'runs/post25'
    if out.exists():raise FileExistsError(out)
    parts=('width','many');src=read(reg);params=read(OUT/'prices.json')
    raw_dirs=sorted(p.name for p in raw.iterdir() if p.is_dir())
    assert not any(p.startswith(parts) for p in raw_dirs),'already started deployment-policy group'
    rows=[];last={}
    for r in src['rows']:
        if r['group'] not in parts:continue
        k=r['group'],r['chip'];c=Contract((0,r['group']),load_chip(r['chip']),'mhc_post','aclnnMhcPost',tuple(r['shape']),'bf16','contiguous_ND',True)
        prior=last.get(k)
        if r['role']=='repeat':assert c==prior
        us,mode=estimate_repeat(c,prior,r['prior_aiv'],params)
        assert abs(us-(r['us'] if r['mode']=='repeat' else r['old_known_us']))<1e-8
        rows.append(dict(chip=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],role=r['role'],state=r['mode'],prior_aiv=r['prior_aiv'],policy_us=us,policy_mode=mode,full_candidate_us=r['us'],public_us=r['old_known_us'],no_history_us=r['old_unknown_us']))
        last[k]=c
    write(out,dict(utc=now(),raw_directories_already_started=raw_dirs,excluded_groups=['boundary','steady'],included_groups=parts,measured=src['measured'],extrapolated=src['extrapolated'],rows=rows,
          expected='5 entirely untouched shapes per SKU, 5 repeat and 5 non-repeat readouts. Assess absolute timing (WAPE/maximum) on repeats and all calls, unknown remains published, DT/PR acceleration; endpoints + predictions fixed in full freeze before any CA.',
          rationale='31 reference windows gave repeat-only generalized improvement. Boundary-group outcomes were seen before this policy freeze; excluded, along with the currently running steady group on both SKUs. Tokens use per-row schedule only when previous successful launch at the same rank, ordered stream, and target had the same full observable installed contract.',
          lifecycle='Reset, forget-history, stream migration or unmodeled submitted work invalidate the key. Any successfully issued different operation replaces the key, including zero-AIV operations. A speculative query creates no state. The scheduler commits a new key after confirmed submission. A purely pre-submit validation failure preserves the preceding successful dispatch; if failure could follow partial submission, invalidate.',
          hashes={str(f.relative_to(ROOT)):sha(f) for f in [reg,OUT/'prices.json',ROOT/'operator_model/post_token_pipeline.py',Path(__file__).resolve(),Path(__file__).with_name('repeat_policy.py')]}))
if __name__=='__main__':main()
