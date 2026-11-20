"""Verify installed runtime files against source and wheel without importing the checkout package."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile


def fingerprint(root,prefix):
    wheel,=(root/'build/wheels/v034').glob('*.whl');files={};sources={}
    with ZipFile(wheel) as z:
        for member in z.namelist():
            if not member.startswith('operator_model/') or member.endswith('/'):continue
            raw=z.read(member);path=member.replace('operator_model/_engine/','operator_model/').replace('operator_model/results/','results/').replace('operator_model/configs/','configs/')
            src=(root/path).read_bytes();expected=src
            if member=='operator_model/__init__.py':
                doc,body=src.split(b'\n',1);expected=doc+b"\nfrom pathlib import Path as _Path\n__path__.append(str(_Path(__file__).with_name('_engine')))\n"+body
            assert expected==raw and (prefix/member).read_bytes()==raw,member
            files[member]={'source':path,'sha256':sha256(raw).hexdigest()};sources[path]=sha256(src).hexdigest()
    return dict(version='0.34.0',baseline='v0.32',wheel=str(wheel.relative_to(root)),wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),runtime_members=len(files),installed_members=files,hashes=sources)
