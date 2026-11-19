"""Run the frozen ordered batch without replacing existing attempts or acquiring extra shapes."""
import fcntl, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.collect import capture,qualify,checked,sha
from development.validation.sku19.campaign import read,write,now
OUT=ROOT/'results/v0.25/post';RAW=ROOT/'runs/post25'
def main():
    reg=OUT/'freeze.json';src=read(reg);RAW.mkdir(exist_ok=True);(OUT/'receipts').mkdir(exist_ok=True)
    for f,h in src['inputs'].items():checked(sha(ROOT/f)==h,'pre-registration hash mismatch '+f)
    with (RAW/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for group in src['groups']:
            for chip in src['measured']:
                ident=group+'-'+chip;folder=RAW/ident;target=OUT/'receipts'/(ident+'.json')
                if target.exists():continue
                jobs=[r for r in src['rows'] if (r['group'],r['chip'])==(group,chip)]
                write(OUT/'status.json',dict(utc=now(),running=ident,count=len(jobs)))
                cost=capture(chip,folder,jobs,src)
                out=dict(chip=chip,group=group,cost=cost,freeze_sha256=sha(reg),rows=[])
                try:
                    checked(cost['exit_code']==0 and not cost['stop_reason'] and cost['started_utc']>src['utc'],'process or chronology')
                    out['rows']=qualify(chip,folder,jobs)
                except (OSError,ValueError,AssertionError,KeyError,IndexError) as err:out['reason']=repr(err)
                out['qualified']=sum(v['timing_qualified'] for v in out['rows']);write(target,out)
                print(ident,out['qualified'],len(jobs),out.get('reason',''),flush=True)
        write(OUT/'status.json',dict(utc=now(),running=None,finished=True))
if __name__=='__main__':main()
