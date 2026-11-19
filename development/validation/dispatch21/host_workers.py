"""Test whether a single CPU simulator worker preserves the published CA contract."""
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,read,write,now,digest
from development.validation.sku19.collect import capture,qualify

OUT=ROOT/'results/v0.21/dispatch'
RAW=ROOT/'runs/dispatch21-host-workers'
CFG=ROOT/'configs/camodel-single-thread.json'


def register():
    origin=read(ROOT/'results/v0.17/evaluation.json')
    paired=next(r for r in origin['rows'] if r['op']=='addmm' and r['shape']==[16,64,256])
    plan=dict(shape=paired['shape'],control={s:paired[s+'_observation'] for s in ('dt','pr')},
        utc=now(),runner_sha=digest(ROOT/'build/smoke/extension_norm_addmm'),config_sha=digest(CFG),
        limit='One host CA worker vs installed default 32. Require same SKU+kernel ELF, active cores, MMAD count, all element checks, same instruction-window duration within one core tick.')
    if (OUT/'host-workers-registration.json').exists():raise FileExistsError('Registered')
    write(OUT/'host-workers-registration.json',plan)


def run():
    plan=read(OUT/'host-workers-registration.json')
    os.environ.update(CAMODEL_CONFIG_PATH=str(CFG),OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    for side in ('pr','dt'):
        chip=CHIPS[side];folder=RAW/side;path=OUT/f'host-workers-{side}.json'
        if path.exists():continue
        shape=plan['shape'];old=plan['control'][side];runner='build/smoke/extension_norm_addmm'
        assert digest(ROOT/runner)==plan['runner_sha'] and digest(CFG)==plan['config_sha']
        j=dict(id='host-workers-addmm',op='addmm',shape=shape,attrs={},kernels=1,nchecks=1,wall_s=420,raw_mib=700,runner=runner,
            args=['addmm',*map(str,shape),'bias'],outputs=[dict(op='addmm',shape=shape,attrs={},indices=[0])])
        cost=read(folder/'cost.json') if (folder/'cost.json').exists() else capture(j,chip,folder)
        try:
            r,=qualify(j,chip,folder)
            configured = re.findall(r'Override esl_top_thread_cnt = (\d+)',(folder/'run.log').read_text())
            st=r['stages'][0];oldstage=old['stages'][0]
            assert configured and set(configured)=={'1'},configured
            assert (st['binary_sha256'],st['active_aic'],st['active_aiv'],st['mmad'])==(oldstage['binary_sha256'],old['active_aic'],old['active_aiv'],oldstage['mmad'])
            same=abs(st['ca_us']-old['ca_us'])<=1/1650+1e-6
            out=dict(qualified=True,same_window=same,baseline_us=old['ca_us'],baseline_trace=old['trace_sha256'],observation=r)
        except (AssertionError,KeyError,OSError,ValueError,IndexError) as exc:
            out=dict(qualified=False,same_window=False,reason=repr(exc))
        write(path,dict(chip=chip,cost=cost,**out))
        print(chip,out.get('reason',out['same_window']),flush=True)


if __name__=='__main__':
    {'register':register,'run':run}[sys.argv[1]]()
