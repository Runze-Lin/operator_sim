"""Prepare unchanged V2 ABI with active output directory redirected for this campaign."""
import os
from pathlib import Path
import subprocess
from build import ROOT, BIN, HERE, compile

GOLD=BIN.parent/'golden'
PYTHON=ROOT/'toolchains/envs/cann-9.2.1/bin/python'
VENDOR=ROOT/'results/chunk-recovery/fla2691/raw/fla_npu/opp/vendors/fla_npu_transformer'
CANN=ROOT/'toolchains/cann/9.2.1/cann-9.2.1'


def prepare():
    old=ROOT/'results/v0.24/kda-prefetch/raw/host-kda.cpp'
    source=old.read_text()
    origin=str(old.parent/'golden')
    assert source.count(origin)==1
    host=BIN/'v2.cpp';host.write_text(source.replace(origin,str(GOLD)))
    api=VENDOR/'op_api'
    compile(host,BIN/'v2',('-I'+str(api/'include'),'-L'+str(api/'lib'),'-Wl,-rpath,'+str(api/'lib'),
                           str(PYTHON.parents[1]/'lib/libstdc++.so.6'),'-lcust_opapi','-ldl'))
    subprocess.run([str(PYTHON),str(HERE.parents[1]/'kda_v23/reference.py'),'1024','12',str(GOLD/'1024-12')],
                   env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},check=True)


if __name__=='__main__':prepare()
