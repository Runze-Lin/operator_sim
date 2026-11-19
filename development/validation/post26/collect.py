"""Serialize extension behind v0.25. Run reserve only for coverage shortfall, regardless of errors."""
import fcntl,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import capture,qualify,sha,checked
from development.validation.sku19.campaign import read,write,now
OUT=ROOT/'results/v0.26/post';OLD=ROOT/'results/v0.25/post';RAW=ROOT/'runs/post26'
def complete_shapes(sku):
    shapes=set()
    for folder,groups in ((OLD,('width','many')),(OUT,('wide','multi','reserve'))):
        for p in (folder/'receipts').glob('*-'+sku+'.json'):
            obj=read(p)
            if obj['group'] not in groups:continue
            for r in obj['rows']:
                if r['timing_qualified'] and r['role']=='repeat' and r['stage']['kernel'].endswith('_6') and r['stage']['active_aiv']==r['active']:shapes.add(tuple(r['shape']))
    return shapes
def run():
    reg=OUT/'registration.json';src=read(reg);RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    for f,h in src['inputs'].items():checked(sha(ROOT/f)==h,'frozen price or collector changed '+f)
    with (RAW/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(OUT/'status.json',dict(utc=now(),state='queued behind post25'))
        while not read(OLD/'status.json').get('finished'):time.sleep(30)
        with (ROOT/'runs/post25/queue.lock').open('a') as slot:
            fcntl.flock(slot,fcntl.LOCK_EX)
            for group in src['order']:
                if group in src['reserve'] and all(len(complete_shapes(chip))>=10 for chip in src['measured']):break
                for chip in src['measured']:
                    ident=group+'-'+chip;dest=OUT/'receipts'/(ident+'.json');folder=RAW/ident
                    if dest.exists():continue
                    jobs=[r for r in src['rows'] if (r['group'],r['chip'])==(group,chip)];write(OUT/'status.json',dict(utc=now(),state='running',id=ident,count=len(jobs)))
                    c=capture(chip,folder,jobs,src);o=dict(chip=chip,group=group,cost=c,freeze_sha256=sha(reg),rows=[])
                    try:
                        checked(c['exit_code']==0 and not c['stop_reason'] and c['started_utc']>src['utc'],'process / chronology');o['rows']=qualify(chip,folder,jobs)
                    except (AssertionError,ValueError,OSError,KeyError,IndexError) as e:o['reason']=repr(e)
                    o['qualified']=sum(x['timing_qualified'] for x in o['rows']);write(dest,o);print(ident,o['qualified'],len(jobs),o.get('reason',''),flush=True)
            write(OUT/'status.json',dict(utc=now(),state='complete',qualified_repeat_shapes={s:len(complete_shapes(s)) for s in src['measured']}))
if __name__=='__main__':run()
