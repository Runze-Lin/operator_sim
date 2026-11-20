"""Match every installed runtime file to the release wheel and its checked-in source."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[3];VERSION='0.32.0'
def digest(b):return sha256(b).hexdigest()
def fingerprint(prefix):
 wheel,=(ROOT/'build/wheels/v032').glob('*.whl');files={};source={}
 with ZipFile(wheel) as z:
  for member in z.namelist():
   if not member.startswith('operator_model/') or member.endswith('/'):continue
   installed=(prefix/member).read_bytes();raw=z.read(member)
   if installed!=raw:raise ValueError('Installed tree differs from frozen wheel: '+member)
   path=member.replace('operator_model/_engine/','operator_model/').replace('operator_model/results/','results/').replace('operator_model/configs/','configs/')
   content=(ROOT/path).read_bytes();source[path]=digest(content);expected=content
   if member=='operator_model/__init__.py':
    doc,body=content.split(b'\n',1);expected=doc+b'\nfrom pathlib import Path as _Path\n__path__.append(str(_Path(__file__).with_name(\'_engine\')))\n'+body
   if expected!=raw:raise ValueError('Source/wheel mismatch: '+path)
   files[member]=dict(source=path,sha256=digest(raw))
 return dict(version=VERSION,baseline='v0.31',wheel=str(wheel.relative_to(ROOT)),wheel_sha256=digest(wheel.read_bytes()),runtime_members=len(files),hashes=source,installed_members=files,packaging={p:digest((ROOT/p).read_bytes()) for p in ('setup.py','pyproject.toml','MANIFEST.in')},contract='All installed runtime data/modules match wheel; source hashes checked byte for byte with only setup.py documented __init__ search-path insertion.')
