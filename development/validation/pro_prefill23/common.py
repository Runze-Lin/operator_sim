from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.23/pro-prefill'
RAW=OUT/'raw'
REG=OUT/'freeze.json'
HOST=ROOT/'build/smoke/large_cast20'
DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
FUTURE='Ascend970DT'
LEGACY=ROOT/'results/v0.20/large'
sys.path.insert(0,str(ROOT))

def when():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(p)
def rel(p):return str(p.relative_to(ROOT))
def state(state,**kwargs):save(OUT/'state.json',dict(utc=when(),state=state,**kwargs))
