"""Test isolated wheel and test-only checkout fixtures without importing checkout API."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import os,subprocess,sys,tempfile,json,re
here=Path(__file__).resolve().parent;root=here.parents[2];prefix=Path(sys.argv[1]).resolve()
assert sys.flags.isolated
with tempfile.TemporaryDirectory(prefix='release32-wheel-tests-') as tmp:
 folder=Path(tmp);(folder/'configs').symlink_to(root/'configs',target_is_directory=True)
 # Some repository tests run a fresh python -m operator_model; its cwd and
 # PYTHONPATH both resolve to the wheel or test fixtures, never checkout code.
 env={**os.environ,'PYTHONPATH':str(prefix)}
 code='import sys,unittest; from pathlib import Path; prefix=Path(sys.argv[1]); root=sys.argv[2]; sys.path.insert(0,str(prefix)); import operator_model; assert Path(operator_model.__file__).is_relative_to(prefix); sys.path.append(root); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover(root+"/tests")); raise SystemExit(not r.wasSuccessful())'
 cmds=[[sys.executable,'-I','-c',code,str(prefix),str(root)],[sys.executable,'-I',str(here/'test_receipts.py')],[sys.executable,'-I',str(here/'test_installed.py'),str(prefix)]]
 out=[]
 for group,c in zip(('full_repository_suite_with_fixtures','receipt_regressions','standalone_wheel_contracts'),cmds):
  p=subprocess.run(c,cwd=folder,env=env,capture_output=True,text=True);output=p.stdout+p.stderr
  if p.returncode:print(output);raise SystemExit(p.returncode)
  match=re.search(r'Ran (\d+) tests? in ([.\d]+)s',output);out.append(dict(group=group,passed=int(match[1]),wall_s=float(match[2])))
 wheel,=(root/'build/wheels/v032').glob('*.whl')
 (root/'results/v0.32/wheel-tests.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),version='0.32.0',wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),groups=out),indent=2)+'\n');print(out)
