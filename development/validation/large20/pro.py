"""DeepSeek-V4-Pro C4s use a separate cast before the 950 indexer."""
import hashlib
import os
import subprocess
import sys
import time
from time import perf_counter
from large_cast import DT, PR, CHIPS, BIN, MODULE, OUT, RAW, ROOT, capture, digest, estimate, now, read, write
from operator_model import load_chip

RECORD=OUT/'pro-freeze.json'
SOURCE='c09d3e20b9c45c80b34ae3781440d90904a44023'
BASE='c057e1efc2148567c4e2e3cefa6281dc62ea58d1'
JOB_ID='cast_262144_64'


def freeze():
    assert not RECORD.exists() and not any(RAW.glob(JOB_ID+'_*'))
    old=read(OUT/'registration.json')
    t,h=262144,64
    src=subprocess.check_output(['git','show',BASE+':operator_model/cast_resource.py'],cwd=ROOT)
    assert hashlib.sha256(src).hexdigest()==old['resource_module']
    ns={'__package__':'operator_model'}
    exec(compile(src,'v0.19/cast_resource.py','exec'),ns)
    role={sku:'independent: CA started after new model freeze' for sku in (DT,PR)}
    old_targets={};new_targets={}
    for chip in CHIPS.values():
        old_us,_,aiv,old_diag=ns['resource_cast']((t,h),load_chip(chip))
        old_targets[chip]=dict(us=old_us,active_aiv=aiv,plan=old_diag['target_plan'])
        start=perf_counter()
        e=estimate('cast_bf16_fp32',(t,h),chip=chip,timing_mode='resource')
        new_targets[chip]=dict(us=e.latency_us,query_ms=(perf_counter()-start)*1000,diagnostics=e.projection['timing_diagnostics'])
    job={**old['jobs'][0], 'id':JOB_ID,'shape':[t,h],'args':[str(t),str(h)],'targets':old_targets}
    write(RECORD, dict(utc=now(),module_sha256=digest(MODULE),runner_sha256=digest(BIN),source_revision=SOURCE,
                        source='deepseek_v4/indexer.py:329-335,738,242 calls A5DeviceAdaptor.prepare_dsa_indexer_weights, device/device_op.py:1588-1590 returns weights.float(). BF16 checkpoint and float Python scalar preserve BF16 input to this op.',
                        job=job,current=dict(id=JOB_ID,shape=[t,h],roles=role,targets=new_targets),
                        queue='One extra CA at a time; wait for load average <220 on 256 cores or for the original trace queue to finish.'))


def collect():
    from large_cast import qualify_one
    reg=read(RECORD)
    assert digest(BIN)==reg['runner_sha256']
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    original=read(OUT/'registration.json')['jobs']
    for c in (DT,PR):
        def done():
            return all((OUT/(j['id']+'_'+s+'.json')).exists() for j in original for s in (DT,PR))
        while os.getloadavg()[0]>220 and not done():
            time.sleep(30)
        qualify_one(reg['job'],c)


if __name__=='__main__':
    {'freeze':freeze,'collect':collect}[sys.argv[1]]()
