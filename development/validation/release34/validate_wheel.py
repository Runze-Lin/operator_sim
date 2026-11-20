"""Test the installed package from a temporary cwd and preserve the test report."""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import os,json,sys,tempfile,subprocess,re

ROOT=Path(__file__).resolve().parents[3];prefix=Path(sys.argv[1]).resolve();assert sys.flags.isolated
with tempfile.TemporaryDirectory(prefix='release34-tests-') as tmp:
    work=Path(tmp);(work/'configs').symlink_to(ROOT/'configs',target_is_directory=True)
    cmd='import sys,unittest; from pathlib import Path; pre=Path(sys.argv[1]); root=sys.argv[2]; sys.path.insert(0,str(pre)); import operator_model; assert Path(operator_model.__file__).is_relative_to(pre) and operator_model.__version__=="0.34.0"; sys.path.append(root); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(root+"/tests")); raise SystemExit(not r.wasSuccessful())'
    env={**os.environ,'PYTHONPATH':str(prefix)};p=subprocess.run([sys.executable,'-I','-c',cmd,str(prefix),str(ROOT)],cwd=work,env=env,text=True,capture_output=True);out=p.stdout+p.stderr
    if p.returncode:print(out);raise SystemExit(p.returncode)
    m=re.search(r'Ran (\d+) tests? in ([.\d]+)s',out);w,=(ROOT/'build/wheels/v034').glob('*.whl')
    v=dict(utc=datetime.now(timezone.utc).isoformat(),version='0.34.0',wheel_sha256=sha256(w.read_bytes()).hexdigest(),isolated_install=str(prefix),groups=[dict(group='full repository suite, installed wheel, fixtures only in checkout',passed=int(m[1]),wall_s=float(m[2]))]);(ROOT/'results/v0.34/wheel-tests.json').write_text(json.dumps(v,indent=2)+'\n');print(v['groups'])
