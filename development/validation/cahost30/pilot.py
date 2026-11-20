"""Measure the two host schedulers against the same installed Matmul and full CA protocol."""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.matmul28 import campaign
from development.validation.matmul28.drain import recover

OUT = ROOT/'results/v0.30/host'
RAW = ROOT/'runs/host30'
REG = OUT/'freeze.json'
CONFIG = ROOT/'configs/camodel-single-thread.json'
SHAPE = [1,2304,1536]
CHIPS = [(campaign.DT,'dt'), (campaign.PR,'pr')]


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    parts = {}
    for chip,tag in CHIPS:
        directory = ROOT/'results/v0.28/matmul'
        stem = '-'.join(map(str,SHAPE))+'-'+tag
        src = directory/(stem+'-drain.json')
        initial = directory/(stem+'.json')
        rec = read(src)
        assert rec['qualified'] and rec['acquisition_sha256']==digest(initial)
        stage, = rec['stages']
        parts[chip] = dict(receipt=str(src.relative_to(ROOT)),receipt_sha256=digest(src),
                           baseline_us=rec['ca_us'],kernel=stage['kernel'],elf=stage['binary_sha256'],
                           roster=sorted(stage['windows']),mmad=stage['mmad'],gold=rec['correctness'],
                           reference_wall_s=read(initial)['cost']['seconds'])
    paths = [Path(__file__).resolve(),CONFIG,campaign.BIN,
             ROOT/'development/validation/matmul28/campaign.py',ROOT/'development/validation/matmul28/drain.py']
    with REG.open('x') as f:
        json.dump(dict(utc=now(),shape=SHAPE,controls=parts,sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
                       host_env=dict(CAMODEL_CONFIG_PATH=str(CONFIG.relative_to(ROOT)),CAMODEL_OMP_THREADS='2',
                                     OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0'),wall_s=480,raw_mib=384,
                       protocol='Input, SKU, msopprof, output checks, release ELF, active physical cores, MMAD, and all-core END+drain match the published measurement. Frequency and simulated blocks unchanged. Each chip at most one 8-minute control. Four physical host CPUs; at least 4 other CPUs under 60% busy available on admission. Two-layer host setting is experimental. Check observed concurrency and keep a separate measurement cohort. Do not label a new operator latency from these controls.'),
                  f,indent=2)
        f.write('\n')


def cpu_ticks():
    cpus={}
    for line in Path('/proc/stat').read_text().splitlines():
        parts=line.split()
        if re.fullmatch(r'cpu\d+',parts[0]):
            counts=list(map(int,parts[1:9]))
            cpus[int(parts[0][3:])]=(sum(counts),sum(counts[3:5]))
    return cpus


def admission(allowed):
    before=cpu_ticks()
    time.sleep(1)
    after=cpu_ticks()
    quiet=[]
    domains=set()
    for cpu in allowed:
        t=after[cpu][0]-before[cpu][0]
        if t and (after[cpu][1]-before[cpu][1])/t>=.4:
            key=(Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()
            if key not in domains:
                quiet.append((1-(after[cpu][1]-before[cpu][1])/t,cpu))
                domains.add(key)
    return [cpu for _,cpu in sorted(quiet)]


def run():
    reg=read(REG)
    assert all(digest(ROOT/name)==want for name,want in reg['sha256'].items())
    RAW.mkdir(parents=True,exist_ok=True)
    with (RAW/'lane.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.nice(9)
        os.environ.update(reg['host_env'],CAMODEL_CONFIG_PATH=str(CONFIG))
        allowed=os.sched_getaffinity(0)
        for chip,tag in CHIPS:
            result=OUT/(tag+'.json')
            if result.exists():
                continue
            ref=reg['controls'][chip]
            assert digest(ROOT/ref['receipt'])==ref['receipt_sha256']
            case='control-'+tag
            folder=RAW/case
            job=dict(shape=SHAPE,wall_s=reg['wall_s'],raw_mib=reg['raw_mib'],
                     predictions={chip:dict(us=ref['baseline_us'])})
            if not (folder/'cost.json').exists():
                if folder.exists():
                    raise RuntimeError(f'Unfinished attempt retained at {folder}')
                while len(quiet:=admission(allowed))<8:
                    time.sleep(15)
                chosen=quiet[4:8]
                os.sched_setaffinity(0,chosen)
                folder.mkdir()
                save(OUT/'state.json',dict(utc=now(),phase='acquiring',chip=chip,host_mask=chosen,pid=os.getpid()))
                campaign.measure(job,chip,folder)
                os.sched_setaffinity(0,allowed)
            cost=read(folder/'cost.json')
            try:
                initial=campaign.qualify(job,chip,folder)
            except (OSError,ValueError,KeyError,IndexError,AssertionError) as exc:
                initial=dict(qualified=False,reason=repr(exc))
            source=OUT/(tag+'-initial.json')
            initial=dict(initial,case=case,shape=SHAPE,chip=chip,cost=cost)
            save(source,initial)
            data=initial
            if not data.get('qualified'):
                try:
                    data=recover(job,initial,folder,acquisition=source) or initial
                except (OSError,ValueError,KeyError,IndexError,AssertionError) as exc:
                    data=dict(qualified=False,reason=repr(exc))
            log=(folder/'run.log').read_text(errors='replace')
            esl=sorted(set(re.findall(r'Override esl_top_thread_cnt = (\d+)',log)))
            concurrency=sorted(set(re.findall(r'parallel esl_sim mode, concurrency num is: (\d+)',log)))
            out=dict(utc=now(),shape=SHAPE,chip=chip,registration_sha256=digest(REG),cost=cost,
                     initial_sha256=digest(source),esl_workers=esl,reported_inner_concurrency=concurrency,
                     kernel_qualified=bool(data.get('qualified')),reason=data.get('reason'),validated_host_control=False)
            if data.get('qualified'):
                s,=data['stages']
                match=(s['kernel']==ref['kernel'] and s['binary_sha256']==ref['elf'] and sorted(s['windows'])==ref['roster']
                       and s['mmad']==ref['mmad'] and data['correctness']==ref['gold'])
                out.update(ca_us=data['ca_us'],baseline_us=ref['baseline_us'],signed_delta_pct=100*(data['ca_us']/ref['baseline_us']-1),
                           identity_and_work_match=match,identical_within_tick=abs(data['ca_us']-ref['baseline_us'])<=1/1650+1e-6,
                           validated_host_control=match and esl==['1'] and concurrency==['2'] and
                             abs(data['ca_us']-ref['baseline_us'])<=1/1650+1e-6,measurement=data)
            save(result,out)
        save(OUT/'state.json',dict(utc=now(),phase='finished',pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','run'))
    {'freeze':register,'run':run}[parser.parse_args().action]()
