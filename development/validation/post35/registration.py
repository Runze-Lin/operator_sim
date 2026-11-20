"""Preprice previously registered, measured-with-the-real-kernel routes; starts no CA."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'results/v0.35/post'
sys.path.insert(0,str(ROOT))
from schedule import schedule

D,P,F='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(ROOT))
def save(p,d,exclusive=False):
    if exclusive and p.exists():raise FileExistsError(p)
    t=p.with_suffix('.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def freeze():
    reg=OUT/'freeze.json'
    if reg.exists():raise FileExistsError(reg)
    specs=[('v0.33/post/freeze.json','v0.33/post/{sku}.json','runs/post33-small/{sku}','channel',[(0,1,208),(1,1,208),(2,3,240),(3,3,240)],'freeze_sha256'),
           ('v0.29/post/split-register.json','v0.29/post/split-{sku}.json','runs/post29-split/{sku}','token_two',[(1,67,4912),(2,69,5680)],'registration_sha256'),
           ('v0.31/post/freeze.json','v0.31/post/mhc_post-{sku}.json','runs/specialized31/post/mhc_post-{sku}','isolated_1024',[(0,1024,5120)],'freeze_sha256')]
    jobs=[]
    for owner,receipt,raw,name,shapes,field in specs:
        for chip in (D,P):
            path=ROOT/'results'/receipt.format(sku=chip);src=ROOT/'results'/owner;workdir=ROOT/raw.format(sku=chip)
            for ordinal,t,d in shapes:
                shape=[t,4,d]
                jobs.append(dict(shape=shape,sku=chip,group=name,ordinal=ordinal,owner=rel(src),owner_sha256=sha(src),owner_field=field,
                    receipt=rel(path),raw=rel(workdir),already_started=workdir.is_dir() or path.is_file(),
                    forecast=schedule(shape,chip),future970_us=schedule(shape,F)['us']))
    fs=[HERE/'schedule.py',Path(__file__),ROOT/'development/validation/post33/candidate.py',ROOT/'results/v0.33/post/prices.json',
        *[ROOT/'operator_model'/n for n in ('post_token_pipeline.py','post_fragment_pipeline.py','post_repeat.py','post_resource.py','post_history.py','rms_resource.py')],
        *[ROOT/('results/'+n) for n in ('v0.25/post/prices.json','v0.28/post/parameters.json','v0.30/post/prices.json','v0.33/post/replay.json')],
        *[ROOT/'configs/chips'/(chip+'.json') for chip in (D,P,F)]]
    save(reg,dict(utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,source_sha256={rel(f):sha(f) for f in fs},
        preflight_sha256=sha(OUT/'preflight.json') if (OUT/'preflight.json').exists() else None,
        public_change=False,selection='resource change for token path; identical v0.33 channel resource model and point rule. Use DT-calibrated per-state row descriptors, full issue overlap and first cost once. No history or probabilities are inputs.',
        independent='only same-contract, same-ordinal reviewed complete vendor instruction windows with actual start after this freeze and identical owner hash; one already-running isolated DT endpoint is replay only, even if no result was available.',
        additional_ca_budget='zero: join the existing small channel, two-fragment and isolated 1024-token collections; no extra devices or host slots.'),True)
if __name__=='__main__':freeze()
