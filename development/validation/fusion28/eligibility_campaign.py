"""One new T/I, first and repeat in each process, serialized after the original four shapes."""
import fcntl
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from campaign28 import OUT, ROOT, DT, PR, FUT, prior, atomic, capture, qualify, sha, now, read
from eligibility import cost
from operator_model import load_chip
from operator_model.sink_registers import price

HERE = Path(__file__).resolve().parent
REG = OUT/'eligibility.freeze.json'
STATE = OUT/'eligibility.state.json'
RAW = ROOT/'runs/sink28-register-group'
ORIGINAL = OUT/'register.json'
T, I = 233, 16


def register():
    if REG.exists() or RAW.exists():
        raise FileExistsError('existing targeted registration/attempt')
    prior.GROUPS = {'paired': [(T, I)]}
    history = prior.novelty()
    for extra in [ORIGINAL, ROOT/'results/v0.26/sink/register.json']:
        assert T not in {row['shape'][0] for row in read(extra)['jobs']}
    assert str(T) not in history
    jobs = []
    for sku in (DT, PR, FUT):
        profile = load_chip(sku)
        h = 0
        for index in range(2):
            a = cost(T, I, sku, h)
            corrected, _, active, diag = price([T, 4], profile, {'num_iters': I})
            assert abs(a['correct_groups_us'] - corrected) < 1e-12
            assert a['active'] == active and diag['target_plan']['register_passes'] == a['g']
            jobs.append(dict(index=index, chip=sku, shape=[T,4], iters=I, prior_h=h,
                             state=a['state'], old_unknown_us=a['old_unknown_us'], corrected_unknown_us=corrected,
                             old_known_us=a['old_known_us'], conditional_us=a['known_us'], stages_us=diag['stages_us'],
                             plan=diag['target_plan'], role='architecture extrapolation' if sku == FUT else 'one forward geometry, two dispatch states'))
            h = active
    assert [jobs[i]['plan']['rows_per_core'] for i in (0,2,4)] == [4,5,2]
    paths = [ROOT/'operator_model/sink_registers.py', ROOT/'operator_model/mhc_sink_resource.py',
             ROOT/'operator_model/vector_v9.py', prior.IMPL, prior.VENDOR, prior.BINARY, HERE/'eligibility_campaign.py',
             ROOT/'development/validation/sink26/collect.py', ROOT/'scripts/cann-run.sh',
             ROOT/'development/validation/rms_ordered_verify/verify.py', ORIGINAL,
             ROOT/'results/v0.9/vector/profile.json', ROOT/'results/v0.26/sink/prices.json', OUT/'prices.json']
    paths.extend(ROOT/'configs/chips'/f'{sku}.json' for sku in (DT, PR, FUT))
    with REG.open('x') as f:
        json.dump(dict(utc=now(), targeted_geometry=[T,4,I], novel_against_archived_tokens=sorted(map(int,history)), jobs=jobs,
                       origin='One discriminating shape with R_DT=4 and R_PR=5. No measurement or training at T=233; the former four-shape, six-window freeze is immutable.',
                       corrected_scope='Replace C=ceil(R/4) by ceil(R/(256B/32B)) only in the steady SIMD charge. Preserve all fixed and other resource prices and legacy T<=256 domain.',
                       acceptance='Qualify both whole calls on each SKU, with all outputs and every AIV retirement. Compare original and corrected on identical windows. DT must match the old price exactly. Publish PR pair WAPE/max, first/repeat signed errors, changes per call and the existing all-history replay; thresholds 25/50. One changed new geometry is not broad coverage.',
                       source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}, budget=dict(mib=2200,wall_s=1200),
                       priority='Only after original four shapes finish, acquire that same exclusive worker.lock; same 50%-CPU, 96-GiB RAM and 120-GiB disk gate. One simulator, no automatic retry, --dump=false.',
                       contract='Released aclnnMhcSinkhorn 4x4 FP32 without intermediates; full output, ELF text identity and complete per-core END, physical AIV roster and 64T written bytes, two ordered calls of identical shape.'),f,indent=2)
        f.write('\n')


def state(label, **fields):
    if STATE.exists():
        old = read(STATE)
        if old['state'] == label and all(old.get(k) == v for k,v in fields.items()):
            return
    atomic(STATE,dict(state=label,utc=now(),registration_sha256=sha(REG),**fields))


def run():
    record=read(REG)
    for p,digest in record['source_hashes'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('registration input changed: '+p)
    while read(OUT/'state.json')['state'] != 'finished':
        original=read(OUT/'state.json')
        state('queued behind original Sinkhorn four-shape campaign',before=original.get('case'),original_state=original['state'])
        time.sleep(45)
    RAW.mkdir(exist_ok=True)
    with (OUT/'raw/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for sku in (DT,PR):
            dest=OUT/f'eligibility.receipt-{sku}.json'
            if dest.exists():continue
            jobs=[j for j in record['jobs'] if j['chip']==sku]
            d=RAW/sku
            if not (d/'cost.json').exists():
                if d.exists():
                    state('interrupted attempt preserved',chip=sku);return
                while True:
                    ram=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])/2**20
                    if os.getloadavg()[0] < .5*len(os.sched_getaffinity(0)) and ram>96 and shutil.disk_usage(ROOT).free>120*2**30:break
                    state('waiting for original CPU, memory and disk headroom',chip=sku);time.sleep(45)
                state('collecting',chip=sku);capture(jobs,d,record['budget'])
            c=read(d/'cost.json');receipt=dict(chip=sku,cost=c,registration_sha256=sha(REG),rows=[])
            try:
                assert c['started_utc']>record['utc']
                rows=qualify(jobs,d,c)
                for j,v in zip(jobs,rows):
                    if not v['qualified']:continue
                    s=v['stage']
                    if (s['active_aiv']!=j['plan']['active_aiv'] or set(s['vf_calls_per_core'].values())!={2*I+2}):
                        v.update(qualified=False,reason='frozen AIV/VF schedule disagrees with observed issue counts')
                receipt['rows']=rows
            except (OSError,AssertionError,ValueError,TypeError,IndexError,KeyError) as exc:
                receipt['reason']=repr(exc)
            atomic(dest,receipt)
            from eligibility import main
            main()
    state('finished')
    from eligibility import main
    main()


if __name__=='__main__':
    {'freeze':register,'collect':run}[sys.argv[1]]()
