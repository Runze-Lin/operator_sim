"""Replay the actual v0.23 public stream in an isolated import root and hash its price inputs."""
from hashlib import sha256
import io,json,os
from pathlib import Path
import subprocess,sys,tarfile,tempfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.24/post'
REV='add9080'
CODE="""
from pathlib import Path
import json,sys
import operator_model
assert operator_model.__version__=='0.23.0'
opened=set()
def audit(event,args):
    if event=='open' and isinstance(args[0],str):
        p=Path(args[0])
        if 'results' in p.parts and p.suffix=='.json' and p!=Path(sys.argv[1]):opened.add(str(p.resolve()))
sys.addaudithook(audit)
jobs=json.loads(Path(sys.argv[1]).read_text())['rows']
streams={}
checked=0;max_delta=0
for r in jobs:
    key=r['chip'],r['group']
    if key not in streams:streams[key]=operator_model.OperatorStream(r['chip'])
    a=streams[key].advance('mhc_post',r['shape'],dtype='bf16')
    b=operator_model.estimate('mhc_post',r['shape'],chip=r['chip'],dtype='bf16')
    diffs=[abs(a.latency_us-r['old_known_us']),abs(b.latency_us-r['old_unknown_us'])]
    assert max(diffs)<1e-10 and a.active_aiv==r['active']
    checked+=1;max_delta=max(max_delta,*diffs)
print(json.dumps(dict(version=operator_model.__version__,ordered_calls=checked,timing_predictions=2*checked,maximum_absolute_diff_us=max_delta,opened_resources=sorted(opened))))
"""
def main():
    revision=subprocess.check_output(['git','rev-parse',REV],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        data=subprocess.check_output(['git','archive',revision,'operator_model','configs/chips'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(data))as archive:archive.extractall(folder,filter='data')
        (folder/'results').symlink_to(ROOT/'results',target_is_directory=True)
        out=subprocess.check_output([sys.executable,'-c',CODE,str(OUT/'freeze.json')],cwd=folder,env={**os.environ,'PYTHONPATH':str(folder)},text=True)
        report=json.loads(out)
    resources={}
    for path in report.pop('opened_resources'):
        f=Path(path);rel=str(f.relative_to(ROOT));blob=subprocess.check_output(['git','show',f'{revision}:{rel}'],cwd=ROOT)
        assert f.read_bytes()==blob
        resources[rel]=sha256(blob).hexdigest()
    (OUT/'baseline-parity.json').write_text(json.dumps(dict(commit=revision,calibration_granted_only_if_prior_blob_identical=resources,**report),indent=2)+'\n')
    print(report)
if __name__=='__main__':main()
