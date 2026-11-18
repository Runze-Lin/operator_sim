"""Run only the full head-pair frontend contract with raw gate and A-log."""
import sys,json,importlib.util
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from operator_model.ca import sha256
freeze=root/'results/v0.8/cube/safe-chunk-freeze.json'
if sys.argv[1]=='freeze':
    if freeze.exists():raise SystemExit('Already frozen')
    cases=[]
    for s,h in [(64,2),(128,2),(64,4),(192,2),(256,2),(128,4),(64,6),(64,8),(192,4),(256,4)]:
        cases.append(dict(id=f'kda_chunk_safe_{s}_{h}',op='kda_chunk',shape=[1,s,h,h,128,128],attrs=dict(safe_gate=True,use_gate_in_kernel=True,lower_bound=-0.03125,chunk_size=64,layout='BSND',dtype='bf16',gate_dtype='fp32',beta_dtype='fp32',state_dtype='fp32'),prediction={'predicted_us':None,'status':'missing qualified fast-path calibration'},query_seconds=None,timeout_s=240,binary_sha256=sha256(root/'build/smoke/cube-v8-kda-safe')))
    files=['scripts/cube-v8-kda-safe.cpp','smoke/extension_kda.cpp','scripts/cube-v8-kda-build.sh','scripts/cube-v8-kda-safe.py','scripts/cube-v8-isolated.py']
    freeze.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,files={f:sha256(root/f) for f in files}),indent=2)+'\n')
else:
    spec=importlib.util.spec_from_file_location('c',root/'scripts/cube-v8-isolated.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    f=json.loads(freeze.read_text());assert all(sha256(root/p)==digest for p,digest in f['files'].items())
    for case in f['cases']:
        result=c.run(case)
        if result['status']!='ok':break
