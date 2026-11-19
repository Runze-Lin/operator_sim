"""Read-only v0.23 resource engine used by the frozen v0.24 experiment.

The package installed by a later release may change. Resolve this experiment's
imports against its small source archive, check every file against the original
prospective registration, and keep the archived JSON rates beside its modules.
"""
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
import sys
from evidence import HERE,ROOT,OUT,read,sha

ARCHIVE=OUT/'v023-resource-sources.zip'
PINS=['operator_model/kda_fla_v2.py','operator_model/chips.py','configs/kda-fla-v2-26.9.1.json']+[
    f'configs/chips/{sku}.json' for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')]


def snapshot():
    reg=read(OUT/'register.json')
    if ARCHIVE.exists():raise FileExistsError(ARCHIVE)
    with ZipFile(ARCHIVE,'x') as z:
        for name in PINS:
            if sha(ROOT/name)!=reg['source_hashes'][name]:raise ValueError('Cannot archive modified '+name)
            z.writestr(ZipInfo(name,date_time=(1980,1,1,0,0,0)),(ROOT/name).read_bytes(),compress_type=ZIP_DEFLATED)


@lru_cache(maxsize=1)
def engine():
    reg=read(OUT/'register.json')
    root=TemporaryDirectory(prefix='operator-sim-kda23-');loc=Path(root.name)
    with ZipFile(ARCHIVE) as z:
        if set(z.namelist())!=set(PINS):raise ValueError('Archive members changed')
        for name in PINS:
            dest=loc/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
            if sha(dest)!=reg['source_hashes'][name]:raise ValueError('Archived SHA differs '+name)
    path=HERE/'model.py'
    if sha(path)!=reg['source_hashes'][str(path.relative_to(ROOT))]:raise ValueError('Frozen experimental model changed')
    package=ModuleType('_operator_sim_archived_kda23')
    package.__path__=[str(loc/'operator_model')];sys.modules[package.__name__]=package
    old=import_module(package.__name__+'.kda_fla_v2');chips=import_module(package.__name__+'.chips')
    aliases={'operator_model.kda_fla_v2':old,'operator_model.chips':chips}
    with patch.dict(sys.modules,aliases),patch.object(sys,'path',sys.path.copy()):
        candidate=run_path(str(path))
    return candidate,root


def predict(shape,chip):
    return engine()[0]['predict'](shape,chip)


if __name__=='__main__':
    if sys.argv[1:]==['snapshot']:snapshot()
    else:
        print(predict((1,129,3,3,128,128),'Ascend950DT_9582')['us'])
