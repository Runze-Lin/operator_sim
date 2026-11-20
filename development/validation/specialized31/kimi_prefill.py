"""Price and measure one wide-output Kimi prefill chunk on the existing B11 CA slot."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.matmul31.device.campaign import ca, recover, triples, RAW as DECODE_RAW

OUT = ROOT/'results/v0.31/matmul/kimi-prefill'
RAW = ROOT/'runs/matmul31/kimi-prefill'
REG = OUT/'freeze.json'
TARGETS = ((ca.DT, 'dt'), (ca.PR, 'pr'))


def register(previous_prefix):
    wide = ca.OUT/'wide-prefill/freeze.json'
    older, = [r for r in ca.read(wide)['rows'] if r['model'] == ca.KIMI]
    assert older['node'] == 'mla.kv_b_proj' and older['shape'] == [262144,3072,512]
    batch, per_request = 8, 4096
    shape = [batch*per_request, *older['shape'][1:]]
    previous = subprocess.check_output([sys.executable, '-I', '-c',
        "import json,sys; sys.path.insert(0,sys.argv[1]); import operator_model as m; "
        "assert m.__version__=='0.30.0'; print(json.dumps([m.estimate('matmul',json.loads(sys.argv[2]),chip=c).latency_us for c in sys.argv[3:]]))",
        previous_prefix,json.dumps(shape),ca.DT,ca.PR,ca.FUT],text=True)
    files = [wide, Path(__file__), Path(ca.__file__), ca.BIN, ca.OUT/'freeze.json']
    registrations = sorted((ROOT/'results').glob('v*/**/freeze.json'))
    for path in [*registrations,ROOT/'results/v0.31/replay.json']:
        if tuple(shape) in triples(ca.read(path)):
            raise ValueError(f'{shape} already registered in {path}')
    prices = {}
    for chip, old in zip((ca.DT,ca.PR,ca.FUT),json.loads(previous)):
        p=ca.estimate('matmul',shape,chip=chip)
        plan=p.projection['target_plan']
        prices[chip]=dict(us=p.latency_us,old_us=old,active_aic=p.active_aic,active_aiv=p.active_aiv,
                          waves=plan['waves'],mmad=plan['mmads_total'],tile=plan['tile'])
    assert not RAW.exists() and not REG.exists()
    OUT.mkdir(parents=True,exist_ok=True)
    ref=ROOT/'results/v0.28/matmul/32768-128-512-dt-drain.json'
    entry=dict(utc=ca.now(),shape=shape,model=ca.KIMI,node=older['node'],phase='prefill',requests=batch,
               tokens_per_request=per_request,scenario='TP8, q=4096 scheduled per request, B8, 32Ki or longer prompt; one 32,768-row step',
               dtype='bf16',contract='A[M,K], B[K,N], contiguous BF16 ND, full golden, CANN aclnnMatmul',
               logical_ops=2*shape[0]*shape[1]*shape[2],logical_tensor_bytes=2*(shape[0]*shape[2]+shape[1]*shape[2]+shape[0]*shape[1]),
               predictions=prices,wall_s=6*3600,raw_mib=24*1024,registration_count=len(registrations),
               installed_matmul_elf=ca.read(ref)['stages'][0]['binary_sha256'],
               dependencies={str(p.relative_to(ROOT)):ca.digest(p) for p in [*files,ref]},
               previous_wheel_sha256=ca.digest(next((ROOT/'build/wheels/v030').glob('*.whl'))),
               wheel_sha256=ca.digest(next((ROOT/'build/wheels/v031').glob('*.whl'))))
    with REG.open('x') as f:
        json.dump(entry,f,indent=2);f.write('\n')
    report()


def report():
    j=ca.read(REG)
    lines=['# Kimi 的多波宽输出 prefill', '',
           'Kimi-K3 的 MLA kv_b_proj 在 TP8 上取 N=3072、K=512。单卡 B8，每请求本步交付 4096 个 token，'
           '得到 [32768,3072,512]。这个形状适用于 32Ki、128Ki、256Ki 请求的其中一个 chunk。'
           '整批 B8×32Ki 的不同几何 [262144,3072,512] 另有原登记。', '',
           '| 芯片 | v0.30 / v0.31 / CA µs | 预测波数 | 预测 / 实测 MMAD | CA 墙钟 min |',
           '|---|---:|---:|---:|---:|']
    for chip, tag in [*TARGETS,(ca.FUT,None)]:
        p=j['predictions'][chip]
        rec=OUT/(tag+'.json') if tag else None
        d=ca.read(rec) if rec and rec.is_file() else {}
        s=d['stages'][0] if d.get('qualified') else {}
        t=f"[{d['ca_us']:.3f}]({rec.name})" if s else '—'
        cost=d.get('cost')
        lines.append(f"| {chip[6:].split('_')[0]}{'（外推）' if not tag else ''} | {p['old_us']:.3f} / {p['us']:.3f} / {t} | "
                     f"{p['waves']} | {p['mmad']} / {s.get('mmad','—')} | {format(cost['seconds']/60,'.1f') if cost else '—'} |")
    lines+=['','逻辑输出为 192 MiB BF16，逐元素核对 100,663,296 个值。每次使用同一个 B11 设备采集槽，'
            '预算为每端 6 小时、24 GiB 原始数据。只有合格回执才有 CA 耗时。'
            '[冻结预测和成本](freeze.json) · [采集状态](state.json) · [整批 prefill](../../../v0.28/matmul/wide-prefill/report.md)','']
    (OUT/'report.md').write_text('\n'.join(lines))


def run():
    j=ca.read(REG)
    assert all(ca.digest(ROOT/name)==h for name,h in j['dependencies'].items())
    RAW.mkdir(parents=True,exist_ok=True)
    os.nice(12)
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    def state(phase,**args):
        ca.save(OUT/'state.json',dict(utc=ca.now(),pid=os.getpid(),phase=phase,**args))
    with (DECODE_RAW/'slot.lock').open('a') as lock:
        state('waiting for B11 slot')
        fcntl.flock(lock,fcntl.LOCK_EX)
        for chip,tag in TARGETS:
            raw=RAW/tag; dest=OUT/(tag+'.json')
            if dest.is_file():
                continue
            if not (raw/'cost.json').is_file():
                if raw.exists() or shutil.disk_usage(ROOT).free < (64*1024+j['raw_mib'])*2**20:
                    raise RuntimeError('Incomplete raw or insufficient disk')
                state('acquiring',chip=chip)
                raw.mkdir()
                ca.measure(j,chip,raw)
            state('qualifying',chip=chip)
            cost=ca.read(raw/'cost.json')
            try:
                assert cost['started_utc']>j['utc'] and not cost['stop_reason'] and cost['exit_code']==0
                try:
                    receipt=ca.qualify(j,chip,raw)
                except (ValueError,AssertionError,KeyError,IndexError,OSError) as exc:
                    initial=OUT/(tag+'-initial.json')
                    d=dict(qualified=False,reason=repr(exc),ca_us=None,case=initial.stem,chip=chip,shape=j['shape'])
                    ca.save(initial,d)
                    receipt=recover(j,d,raw,acquisition=initial) or d
                if receipt['qualified']:
                    st,=receipt['stages']
                    assert st['binary_sha256']==j['installed_matmul_elf']
                    forecast=j['predictions'][chip]
                    receipt['mechanism_checks']={key:st[key]==forecast[key] for key in ('active_aic','active_aiv','mmad')}
            except (ValueError,AssertionError,KeyError,IndexError,OSError) as exc:
                receipt=dict(qualified=False,ca_us=None,reason=repr(exc))
            ca.save(dest,dict(receipt,shape=j['shape'],chip=chip,cost=cost,freeze_sha256=ca.digest(REG)))
            report()
        state('finished')


if __name__=='__main__':
    if sys.argv[1]=='freeze':
        register(sys.argv[2])
    elif sys.argv[1]=='run':
        run()
    else:
        report()
