"""Verify finalized official multi-launch exports, which CA moves out of its device0 staging directory."""
from hashlib import sha256
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'development/validation/rms_ordered_verify'))
from development.validation.rms_unknown23.collect import REG,RAW,OUT,host,verify,read,write


def assess(src,original):
    chip=original['chip'];group=original['process'];jobs=[j for j in read(REG)['rows'] if j['chip']==chip and j['process']==group];folder=RAW/src.stem;cost=read(folder/'cost.json');dest=OUT/'verified'/src.name
    record={**original,'rows':[],'qualified':False,'original_qualification_sha256':verify.digest(src)}
    record.pop('reason',None)
    try:
        verify.checked(cost==original['cost'] and cost['exit_code']==0 and not cost['stop_reason'],'capture exit/cost')
        verify.checked('Profiling data parse finished.' in (folder/'run.log').read_text(),'msOpProf export not finished')
        verify.checked((folder/'inputs.txt').read_text()==''.join(f"rmsnorm {j['i']} {j['shape'][0]} {j['shape'][1]} 0\n" for j in jobs),'actual input contracts')
        tasks,tables=host(folder,chip,jobs);elf=verify.text_bytes(verify.VENDOR)
        profile,=folder.glob('profile/OPPROF_*');published=profile/verify.KERNEL
        if not published.is_dir():published=profile/'device0'/verify.KERNEL
        verify.checked(sorted(int(x.name) for x in published.iterdir() if x.is_dir())==list(range(len(jobs))),'ordinal directories')
        record.update(host_numeric_groups=len(tasks)*2,elf_text_sha256=sha256(elf).hexdigest(),log_sha256=verify.digest(folder/'run.log'),inputs_sha256=verify.digest(folder/'inputs.txt'))
        for j,task,tab in zip(jobs,tasks,tables):
            try:
                r=verify.launch(published/str(j['i']),chip,j,task,tab,elf);verify.checked(r['active_count_matches'],'physical AIV mismatch')
            except (ValueError,KeyError,OSError,IndexError,TypeError) as error:
                r=dict(**j,timing_qualified=False,stage=None,reason=repr(error),correctness=task['golden'])
            record['rows'].append(r)
        good=[r['stage'] for r in record['rows'] if r['timing_qualified']];verify.checked(all(b['first_tick']>a['last_tick'] for a,b in zip(good,good[1:])),'time order');record['qualified']=len(good)==len(jobs)
    except (ValueError,KeyError,OSError,IndexError,TypeError) as err:record['reason']=repr(err)
    write(dest,record);print(src.stem,record['qualified'],[(r['i'],r.get('reason')) for r in record['rows'] if not r['timing_qualified']],record.get('reason',''),flush=True)


def update():
    (OUT/'verified').mkdir(exist_ok=True)
    for src in sorted((OUT/'receipts').glob('*.json')):
        dest=OUT/'verified'/src.name
        if not dest.is_file():assess(src,read(src))
    from development.validation.rms_unknown23.report import independent
    independent()

if __name__=='__main__':
    while True:
        update()
        if 'watch' not in sys.argv or read(OUT/'status.json')['state']=='complete':break
        time.sleep(12)
