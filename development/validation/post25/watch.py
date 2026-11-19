"""Update evidence only when a new immutable receipt appears. Never start a new CA."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read
OUT=ROOT/'results/v0.25/post';CODE=Path(__file__).resolve().parent

def run(name,*args):
    ans=subprocess.run([sys.executable,str(CODE/name),*args],cwd=ROOT,check=False)
    if ans.returncode:print(name,'failed',ans.returncode,flush=True)
def main():
    last=-1;attempted=set()
    while True:
        n=len(list((OUT/'receipts').glob('*.json')))
        if n!=last:
            for group,t,d,slug in [('boundary',67,384,'two-token-repeat'),('many',193,896,'multi-token-repeat')]:
                if group in attempted or not (OUT/'receipts'/(group+'-Ascend950DT_9582.json')).exists():continue
                if not (OUT/(slug+'.png')).exists():run('figure.py',group,str(t),str(d),slug)
                attempted.add(group)
            run('milestones.py');run('report.py');last=n
            print('updated from',n,'completed processes',flush=True)
        if n==8:break
        time.sleep(25)
if __name__=='__main__':main()
