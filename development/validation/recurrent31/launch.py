"""Start only the frozen B6 pair; B128 is never launched here."""
from pathlib import Path
import os,subprocess
root=Path(__file__).resolve().parents[3]
state=root/'runs/recurrent31';state.mkdir(parents=True,exist_ok=True)
# Existing large29 slot serializes the jobs. Use the same reserved CPU set
# selected by that supervisor, not a narrow per-core affinity during CA.
import json
cpus=set(json.loads((root/'results/v0.29/large/cube/host.json').read_bytes())['logical_cpu_ids'])
if not cpus <= os.sched_getaffinity(0):raise RuntimeError('Registered CPUs unavailable on this host')
with (state/'supervisor.log').open('a') as log:
    p=subprocess.Popen([str(root/'toolchains/envs/cann-9.2.1/bin/python'),str(Path(__file__).with_name('check.py')),'run'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,cpus))
    print(p.pid,sorted(cpus))
