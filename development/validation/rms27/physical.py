"""Score a distinct, all-ELF-END cohort without synthetic labels, on the original captures alone."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms27.campaign import OUT,REG,RAW,host,verify,read,write
# The v0.24 standalone script imports its verification helper via an unqualified
# 'campaign' module. Load its exact function with the same verifier dependency.
from importlib.util import module_from_spec,spec_from_file_location
from types import SimpleNamespace
from unittest.mock import patch
spec=spec_from_file_location('rms_physical24',ROOT/'development/validation/rms_prefix24/physical.py')
physical24=module_from_spec(spec)
with patch.dict(sys.modules,{'campaign':SimpleNamespace(verify=verify)}):
    spec.loader.exec_module(physical24)
measure=physical24.measure
from development.validation.rms27.limits import group
from development.validation.rms27.report import summarize

def run():
    reg=read(REG);observations=[];failed=[];audit=OUT/'physical-recovery';audit.mkdir(exist_ok=True)
    for f in (OUT/'receipts').glob('*.json'):
        r=read(f);g=r['group'];sku=r['chip'];folder=RAW/f.stem;trigger=None
        for v in r['rows']:
            if v['role']=='prefix':continue
            if v['timing_qualified']:
                z=v['stage'];duration=z['physical_ca_us'];trace=z['trace'];origin='verified all-core END on strict capture'
            else:
                dest=audit/(f.stem+f"-{v['i']}.json")
                if not dest.exists():
                    try:
                        if trigger is None:
                            jobs=[j for j in reg['rows'] if (j['chip'],j['group'])==(sku,g)]
                            tasks,tabs=host(folder,sku,jobs);prof,=folder.glob('profile/OPPROF_*');pub=prof/verify.KERNEL
                            if not pub.is_dir():pub=prof/'device0'/verify.KERNEL
                            trigger=(tasks,tabs,pub,verify.text_bytes(verify.VENDOR))
                        tasks,tabs,pub,elf=trigger;d=pub/str(v['i'])
                        physical=measure(d,sku,v,tabs[v['i']],elf)
                        missing=[]
                        for sh in (d/'simulator').glob('core*/trace.json'):
                            if not any(x['name']=='END_LABEL' for x in verify.events(sh)):missing.append(sh.parent.name)
                        result=dict(qualified=True,correctness=tasks[v['i']]['golden'],missing_labels=missing,**physical)
                    except (ValueError,OSError,TypeError,IndexError,KeyError) as e:result=dict(qualified=False,reason=repr(e))
                    write(dest,result)
                result=read(dest)
                if not result['qualified']:failed.append(dict(shape=v['shape'],sku=sku,role=v['role'],reason=result['reason']));continue
                duration=result['ca_us'];trace=result['trace'];origin=str(dest.relative_to(ROOT))
            observations.append(dict(**{k:v[k] for k in ('shape','chip','role','prior_aiv','active_aiv','known_us','us','old_us','cold_hot_us')},ca_us=duration,trace=trace,origin=origin))
    complete=[v for v in observations if any((q['chip'],q['shape'])==(v['chip'],v['shape']) and q['role']!=v['role'] for q in observations)]
    out=dict(scope='All physical ENDs from ORIGINAL captures, no synthetic label on any lane. All qualified rows and goldens/ABI belong to their original exact same host process. No same-prefix retry in these metrics.',rows=observations,failed=failed,summary=summarize(complete),bounds=group(complete,reg['measured']))
    write(OUT/'physical.json',out);return out
if __name__=='__main__':run()
