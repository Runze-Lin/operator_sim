"""Print completed CA counts and keep cumulative trace summaries up to date."""
import json,time,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
expected=json.loads((root/'results/v0.8/fusion/freeze.json').read_text())['cases']
last=None
while True:
    count=sum((root/'runs/v8cv'/c['id']/'result.json').exists() for c in expected)
    if count!=last:
        subprocess.run([sys.executable,str(root/'scripts/fusion-v8-report.py')],check=True,cwd=root)
        print(f'{count}/{len(expected)} native validation cases collected',flush=True);last=count
    if count==len(expected):break
    time.sleep(10)
