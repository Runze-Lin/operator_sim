"""Refresh only when a registered device receipt arrives; never starts device work."""
from pathlib import Path
from hashlib import sha256
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];O=R/'results/v0.34/attn'
def watch():
    frozen=json.loads((O/'freeze.json').read_bytes());files=[]
    for j in frozen['jobs']:
        if j['new']:
            files.extend(O/(f"{j['shape'][0]}-{j['shape'][3]}-{tag}.json") for tag in ('dt','pr'))
        else:
            files.extend(R/p for p in j['existing'])
    def identities():
        return [(str(p),sha256(p.read_bytes()).hexdigest() if p.is_file() else None) for p in files]
    previous=identities()
    deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline and not all(h for p,h in previous):
        time.sleep(45);state=identities()
        if state!=previous:
            subprocess.run([sys.executable,str(HERE/'report.py')],cwd=R,check=True)
            previous=state
if __name__=='__main__':watch()
