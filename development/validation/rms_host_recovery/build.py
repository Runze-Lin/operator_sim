"""Build the v0.21 retry-only host from the original with a single gate substitution."""
from pathlib import Path
import sys
import subprocess
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku19.campaign import digest, read

BUILD = ROOT/'build/rms-host-recovery'
RUNNER = BUILD/'large_vector_rms16'
SOURCE = BUILD/'large_vector_rms16.cpp'
ORIGINAL = ROOT/'smoke/large_vector.cpp'
OLD_GATE = 'c<=0 || c>7168 || c%64'
NEW_GATE = 'c<=0 || (op=="rmsnorm" ? (c>8192 || c%16) : (c>7168 || c%64))'


def build():
    freeze=read(ROOT/'results/v0.20/rms/freeze.json')
    assert digest(ORIGINAL)==freeze['refs']['smoke/large_vector.cpp']
    source=ORIGINAL.read_text()
    assert source.count(OLD_GATE)==1
    revised=source.replace(OLD_GATE,NEW_GATE)
    BUILD.mkdir(exist_ok=True)
    if SOURCE.exists() and SOURCE.read_text()!=revised:
        raise FileExistsError(SOURCE)
    SOURCE.write_text(revised)
    # Identical include, library and optimization flags as sku19/build-large.sh.
    cmd = '''g++ -std=c++17 -O2 -Wall -Wextra "$1" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$2"'''
    subprocess.run(['bash',ROOT/'scripts/cann-run.sh','bash','-c',cmd,'build',SOURCE,RUNNER],check=True)


if __name__ == '__main__':
    build()
