"""Unpack only the signed-off FLA runtime, preserving published bytes without installing it."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import hashlib
import json
import stat

from probe import RAW

def main():
    release = json.loads((RAW/'release.json').read_bytes())
    [asset] = [a for a in release['assets'] if a['name']=='flash_linear_attention_npu_a5-26.9.1-py3-none-manylinux_2_34_aarch64.whl']
    wheel=RAW/'fla-a5.whl'
    if wheel.stat().st_size!=asset['size'] or 'sha256:'+hashlib.sha256(wheel.read_bytes()).hexdigest()!=asset['digest']:
        raise ValueError('wheel differs from release asset')
    with ZipFile(wheel) as z:
        for item in z.infolist():
            if item.is_dir() or '/purelib/fla_npu/' not in item.filename or '/offline/' in item.filename:
                continue
            relative=PurePosixPath(item.filename.split('/purelib/',1)[1])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0]!='fla_npu' or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('unexpected archive member '+str(relative))
            path=RAW/Path(relative)
            if path.exists():
                if path.read_bytes()!=z.read(item):raise ValueError('changed extracted release file '+str(relative))
                continue
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(item))

if __name__=='__main__':main()
