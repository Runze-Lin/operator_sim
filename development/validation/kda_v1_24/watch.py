"""Publish diagnostics once each frozen CA receipt is atomically visible."""
import subprocess
import sys
import time
from investigate import HERE, REG, OUT, ROOT, read, save, when
sys.path.insert(0,str(HERE))
from inspect_sync import inspect
from report import summary

def main():
    cases=read(REG)['cases'];conda=ROOT/'toolchains/envs/cann-9.2.1/bin/python'
    for c in cases:
        receipt=OUT/(c['id']+'.json')
        while not receipt.exists():time.sleep(12)
        if not read(receipt).get('skipped'):
            try:
                inspect(c)
                subprocess.run([str(conda),'-c','from numerical_details import replay;from investigate import REG,read;import sys;replay(next(c for c in read(REG)["cases"] if c["id"]==sys.argv[1]))',c['id']],cwd=HERE,check=True)
            except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
                save(OUT/(c['id']+'-diagnostic-error.json'),dict(utc=when(),error=str(exc)))
        summary()
    subprocess.run([str(conda),str(HERE/'operand_replay.py')],check=True)
    if all((OUT/(sku+'-t2-h1-varlen-sync.json')).exists() for sku in ['dt','pr']):
        subprocess.run([sys.executable,str(HERE/'compare_pair.py')],check=True)
    summary()
    print('all frozen receipts analyzed',flush=True)
if __name__=='__main__':main()
