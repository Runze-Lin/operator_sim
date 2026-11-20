"""Refresh installed-release scores only when a final endpoint file is added or replaced."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.34';RAW=ROOT/'runs/release34';HERE=Path(__file__).parent.resolve()
ROSTERS=[('v0.34/addmm','[dp][tr].json'),('v0.34/attn','[0-9]*-??.json'),('v0.34/indexer','Ascend950*.json'),('v0.34/recurrent','[dp][tr].json'),
         ('v0.32/matmul','*drain.json'),('v0.32/matmul/decode','[0-9]*-??.json'),('v0.32/matmul/pro-large','[dp][tr].json'),('v0.32/matmul/kimi-head','[dp][tr].json'),
         ('v0.30/matmul-dispatch/device','[0-9]*-??.json'),('v0.28/matmul/wide-prefill','*-??.json'),('v0.28/matmul/narrow','*-??.json'),('v0.28/matmul/long-prefill','*-??.json'),
         ('v0.29/large/vector','instruction-*.json'),('v0.29/large/cube','*-[dp][tr].json'),('v0.29/large/fusion','*-Ascend950*.json'),('v0.29/large/fa','fa-*.json'),
         ('v0.33/post','Ascend950*.json'),('v0.31/post','*Ascend950*.json'),('v0.29/post','split-Ascend950*.json')]

def fingerprints():
    return {str(p.relative_to(ROOT)):p.stat().st_mtime_ns for d,pattern in ROSTERS for p in sorted((ROOT/'results'/d).glob(pattern))}

def main(prefix):
    RAW.mkdir(exist_ok=True);path=RAW/'roster.json'
    with (RAW/'observer.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last=json.loads(path.read_bytes()) if path.exists() else None
        while True:
            current=fingerprints()
            if last!=current:
                command=[sys.executable,'-I',str(HERE/'evaluate.py'),str(prefix)];env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
                p=subprocess.run(command,cwd='/tmp',env=env)
                if p.returncode:raise SystemExit(p.returncode)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(current)+'\n');tmp.replace(path);last=current
            current_targets=[OUT/f'{op}/{chip}.json' for op in ('addmm','recurrent') for chip in ('dt','pr')]
            current_targets += [OUT/f'indexer/{chip}.json' for chip in ('Ascend950DT_9582','Ascend950PR_957b')]
            current_targets += [OUT/f'attn/{stem}-{chip}.json' for stem in ('65-1','121-2') for chip in ('dt','pr')]
            if all(p.exists() for p in current_targets):return
            time.sleep(45)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('prefix',type=Path);args=parser.parse_args();main(args.prefix.resolve())
