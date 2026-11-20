"""Blind prices for existing queue slots: never open any ongoing trace or latency."""
from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model.matmul_supply32 import resource_matmul_dispatch as price
from operator_model.matmul_dispatch31 import resource_matmul_dispatch as old
from operator_model import load_chip

OUT=ROOT/'results/v0.32/matmul'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
ROUTES=[
    ([524288,128,512],'Flash B8×64Ki','v0.30/matmul-dispatch/device','524288','runs/matmul_dispatch30/device'),
    ([2097152,128,512],'Flash B8×256Ki','v0.28/matmul/long-prefill','262144','runs/matmul28/long-prefill'),
    ([32768,3072,512],'Kimi B8×4096 chunk','v0.31/matmul/kimi-prefill','','runs/matmul31/kimi-prefill'),
    ([262144,3072,512],'Kimi B8×32Ki','v0.28/matmul/wide-prefill','Kimi-K3','runs/matmul28/wide-prefill'),
]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def freeze():
    rows=[]
    for shape,desc,archive,stem,folder in ROUTES:
        for sku,tag in ((DT,'dt'),(PR,'pr'),(FUT,None)):
            chip=load_chip(sku); old_us=old(shape,chip)[0];us,ac,av,diag=price(shape,chip);d=diag['target_plan'];qual='970DT extrapolation';ticket=None
            if tag:
                # Inspect directory existence only. No simulator output of any new endpoint is used.
                variants=[f'{stem}-{tag}'] if stem else [tag,sku]
                raw=ROOT/folder; dirs=[raw/v for v in variants]
                predir=ROOT/'results'/archive
                tickets=[predir/(a+post+'.json') for a in variants for post in ('-retired','-drain','')]
                ticket=[str(t.relative_to(ROOT)) for t in tickets]
                if any(t.exists() for t in tickets):qual='already received before v0.32 freeze; replay only'
                elif any(a.exists() for a in dirs):qual='running before freeze; result blinded'
                else:qual='prediction before device acquisition'
            rows.append(dict(shape=shape,scenario=desc,chip=sku,role=qual,candidate_receipts=ticket,old_us=old_us,us=us,aic=ac,aiv=av,plan=d,pricing=diag.get('pricing'),stages_us=diag['stages_us']))
    deps=['operator_model/matmul_supply32.py','operator_model/matmul_v15.py','operator_model/matmul_resource.py','operator_model/matmul_supply30.py','operator_model/matmul_dispatch31.py','operator_model/matmul_waves.py','operator_model/matmul_splitk.py','results/v0.16/matmul.json','results/v0.30/matmul/prices.json','development/validation/matmul32/freeze.py',*(f'configs/chips/{s}.json' for s in (DT,PR,FUT))]
    with (OUT/'freeze.json').open('x') as f:json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,sha256={s:digest(ROOT/s) for s in deps},policy='No new CA jobs; reuse existing slots and budgets. The 32Ki and 128Ki Flash DT/PR traces informed development. 64Ki and 256Ki are untouched holdouts; distinguish already running from preregistered. Nominal width case outside B residency retains v0.31.'),f,indent=2,ensure_ascii=False);f.write('\n')
if __name__=='__main__':freeze()
