"""Rebuild scores only after a registered terminal receipt appears or changes."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;R=HERE.parents[2];O=R/'results/v0.35/post'
reg=json.loads((O/'freeze.json').read_bytes())
files={R/j['receipt'] for j in reg['jobs']}|{O/'fresh-dt.json',O/'fresh-pr.json'}
def changes():return {p:hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in files}
previous=changes()
# Catch a receipt deposited in the short time since the report was written.
if any(p.is_file() and p.stat().st_mtime>(O/'forward.json').stat().st_mtime for p in files):
    subprocess.run([sys.executable,str(HERE/'evaluate.py')],cwd=R,check=True)
deadline=time.monotonic()+72*3600
while time.monotonic()<deadline and not all(previous.values()):
    time.sleep(30);current=changes()
    if current!=previous:
        subprocess.run([sys.executable,str(HERE/'evaluate.py')],cwd=R,check=True);previous=current
