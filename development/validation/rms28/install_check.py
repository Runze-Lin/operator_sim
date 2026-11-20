"""Build and import the current wheel from an isolated directory; compare frozen prices."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.rms28.campaign import OUT,read,write
SCRIPT="""import json,sys
from pathlib import Path
from time import perf_counter_ns
sys.path.insert(0,sys.argv[1])
import operator_model
from operator_model import estimate
from operator_model.rmsnorm_pipeline import rmsnorm_pipeline
assert Path(operator_model.__file__).resolve().is_relative_to(Path(sys.argv[1]))
frozen=json.loads(Path(sys.argv[2]).read_text())['rows']
public=[];known=[];specific=[];runtime=[]
for r in frozen:
    kw=dict(chip=r['chip'])
    t=perf_counter_ns();default=estimate('rmsnorm',r['shape'],**kw);runtime.append((perf_counter_ns()-t)/1e6)
    public.append(abs(default.latency_us-r[sys.argv[3]]))
    explicit=estimate('rmsnorm',r['shape'],{'resident_aiv':r['prior_aiv']},**kw)
    known.append(abs(explicit.latency_us-r['known_us']))
    specialized=rmsnorm_pipeline(r['shape'],**kw,resident_aiv=r['prior_aiv'],immediate_repeat=r['role']=='repeat',diagnostics=True)
    specific.append(abs(specialized['us']-r['specific_context_us']))
assert max(*public,*known,*specific)<1.e-10
print(json.dumps(dict(version=operator_model.__version__,rows=len(frozen),public_expected=sys.argv[3],
  public_max_abs_delta_us=max(public),known_public_max_abs_delta_us=max(known),
  specialized_inspect_max_abs_delta_us=max(specific),public_worst_with_first_import_ms=max(runtime),
  exposed_at_package_root=hasattr(operator_model,'rmsnorm_pipeline') and callable(operator_model.rmsnorm_pipeline))))
"""
def run(expected):
    with tempfile.TemporaryDirectory(prefix='rms28-package-') as tmp:
        work=Path(tmp)
        src=work/'source'; src.mkdir()
        for name in ('MANIFEST.in','pyproject.toml','setup.py','README.md'):
            shutil.copy2(ROOT/name,src/name)
        shutil.copytree(ROOT/'operator_model',src/'operator_model',ignore=shutil.ignore_patterns('__pycache__'))
        for line in (src/'MANIFEST.in').read_text().splitlines():
            if line.startswith(('include results/','include configs/')):
                name=line.split()[1]
                (src/name).parent.mkdir(exist_ok=True,parents=True)
                shutil.copy2(ROOT/name,src/name)
        env=dict(os.environ,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','-w',str(work),str(src)],
            env=env,cwd=work,check=True,capture_output=True)
        wheel,=work.glob('*.whl')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(work/'installed'),str(wheel)],
            env=env,cwd=work,check=True,capture_output=True)
        output=subprocess.check_output([sys.executable,'-I','-c',SCRIPT,str(work/'installed'),str(OUT/'freeze.json'),expected],
            env=env,cwd=work,text=True)
        write(OUT/'installed-wheel.json',json.loads(output))
if __name__=='__main__': run(sys.argv[1])
