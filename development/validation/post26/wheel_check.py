"""Install a wheel in an isolated temporary tree and query its Post data from outside checkout."""
import json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post26.freeze import OUT
from development.validation.sku19.campaign import write
from hashlib import sha256

def main():
    with tempfile.TemporaryDirectory() as tmp:
        w=Path(tmp);project=w/'project';project.mkdir();wheel_dir=w/'wheels'
        for f in ('README.md','pyproject.toml','setup.py','MANIFEST.in'):shutil.copy2(ROOT/f,project/f)
        shutil.copytree(ROOT/'operator_model',project/'operator_model',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for line in (ROOT/'MANIFEST.in').read_text().splitlines():
            if not line.startswith(('include results/','include configs/')):continue
            f=line.split(maxsplit=1)[1];(project/f).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,project/f)
        pr=subprocess.run([sys.executable,'-m','pip','wheel','--no-index','--no-build-isolation','--no-deps','-w',str(wheel_dir),'.'],cwd=project,capture_output=True,text=True)
        if pr.returncode:raise ValueError(pr.stderr)
        whl,=wheel_dir.glob('*.whl');dest=w/'installed';wh=zipfile.ZipFile(whl);wh.extractall(dest);resource='operator_model/results/v0.25/post/prices.json'
        assert sha256(wh.read(resource)).hexdigest()==sha256((ROOT/'results/v0.25/post/prices.json').read_bytes()).hexdigest()
        code="""from operator_model import OperatorStream
import json
ans={}
for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
 s=OperatorStream(sku,experimental_post_repeats=True)
 out=[s.advance('mhc_post',[65,4,5120]) for _ in range(2)];r=out[-1]
 ans[sku]=dict(first_us=out[0].latency_us,repeat_us=r.latency_us,model=r.projection['timing_diagnostics']['model'],fragment=r.projection['timing_diagnostics']['channel_tiles'])
 assert ans[sku]['model']=='post_token_immediate_repeat'
print(json.dumps(ans))"""
        v=subprocess.run([sys.executable,'-c',code],cwd=w,env={**os.environ,'PYTHONPATH':str(dest)},check=True,capture_output=True,text=True)
        write(OUT/'wheel-check.json',dict(distribution=whl.name,mode='isolated build, unpacked wheel imported from outside checkout, no CANN calls',prices_sha256=sha256(wh.read(resource)).hexdigest(),results=json.loads(v.stdout)))
if __name__=='__main__':main()
