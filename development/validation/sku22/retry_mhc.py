"""One serial full-window remeasurement of specific irrecoverable Post endpoints."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(HERE))
from verify_mhc import digest, verify, write
from development.validation.sku19.campaign import now, read

OUT = ROOT/'results/v0.22/recovered'
REG = OUT/'retry'/'registration.json'
RAW = ROOT/'build/mhc-window-recovery'
ID = 'mhc_post_253_4_2048'
CHIPS = ('Ascend950DT_9582','Ascend950PR_957b')


def register(archive):
    if REG.exists() or RAW.exists():
        raise FileExistsError(REG)
    freeze=archive/'results/v0.21/dispatch/freeze.json'
    job,=[j for j in read(freeze)['jobs'] if j['id']==ID]
    assert digest(archive/job['runner'])==job['runner_sha256']
    entries=[]
    for chip in CHIPS:
        archived=OUT/(ID+'_'+chip+'.json')
        evidence=read(archived)
        assert not evidence['qualified'] and 'missing/duplicate physical END' in evidence['reason']
        entries.append(dict(chip=chip,original=evidence['original'],original_sha256=evidence['original_sha256'],
                            missing_end=evidence['reason'],job=job))
    REG.parent.mkdir(parents=True,exist_ok=True)
    write(REG,dict(utc=now(),original_freeze=digest(freeze),collector_sha256=digest(Path(__file__)),
        collector_backend_sha256=digest(archive/'development/validation/sku19/collect.py'),
        archive_root=str(archive),entries=entries,concurrency=1,wall_s=3600,raw_mib=8192,
        reason='No raw executed/popped logs in either original Post case. Per-core source shards and merged JSON lose the same physical END on DT five AIVs and PR one AIV. Vendor durations are rounded and cannot supply exact completion ticks. Original host inputs, outputs and predictions unchanged.',
        qualification='Clean full process, two named inputs/subscope matching the registered host, exact runtime SKU and memory, all-element golden, correct package ELF, official all-core windows and every physical END required. One capture per previously ineligible endpoint.'))


def run(archive):
    reg=read(REG)
    assert reg['archive_root']==str(archive) and reg['collector_sha256']==digest(Path(__file__))
    collector=archive/'development/validation/sku19/collect.py'
    assert digest(collector)==reg['collector_backend_sha256']
    spec=importlib.util.spec_from_file_location('original_mhc_capture',collector)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    RAW.mkdir(exist_ok=True)
    with (RAW/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in reg['entries']:
            source=archive/item['original']
            assert digest(source)==item['original_sha256']
            chip=item['chip']
            dest=OUT/source.name
            prior=read(dest)
            old_retry=source.parent.parent/'retries'/source.name
            if prior['qualified'] or read(source)['qualified'] or old_retry.exists() and read(old_retry)['qualified']:
                continue
            result=REG.parent/source.name
            if result.exists():
                continue
            folder=RAW/source.stem
            job={**item['job'],'wall_s':reg['wall_s'],'raw_mib':reg['raw_mib']}
            assert digest(archive/job['runner'])==job['runner_sha256']
            write(REG.parent/'status.json',dict(utc=now(),state='running',endpoint=source.stem,raw=str(folder.relative_to(ROOT))))
            if (folder/'cost.json').is_file():
                cost=read(folder/'cost.json')
            elif folder.exists():
                raise RuntimeError(f'Persist interrupted capture for audit: {folder}')
            else:
                cost=m.capture(job,chip,folder)
            record={k:prior[k] for k in ('shape','chip','op','original','original_sha256')}
            record.update(new_ca_seconds=cost['seconds'],registration=str(REG.relative_to(ROOT)),original_failure=prior['reason'],
                       raw_root='v0.22 checkout',cost=cost)
            try:
                if cost['stop_reason'] or cost['exit_code'] or cost['started_utc']<=reg['utc']:
                    raise ValueError('new process/budget/chronology')
                log=(folder/'run.log').read_text()
                if 'Profiling data parse finished.' not in log or 'All task success.' not in log:
                    raise ValueError('profiler did not complete')
                record.update(qualified=True,observation=verify(archive,source,capture_folder=folder))
            except (ValueError,OSError,KeyError,StopIteration) as exc:
                record.update(qualified=False,reason=repr(exc))
            write(result,record)
            write(dest,record)
            print(dest.stem,record['qualified'],record.get('reason',''),flush=True)
        write(REG.parent/'status.json',dict(utc=now(),state='complete',outcomes={p.stem:read(p)['qualified'] for p in REG.parent.glob('mhc*.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('register','run'));parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    {'register':register,'run':run}[args.action](args.archive.resolve())
