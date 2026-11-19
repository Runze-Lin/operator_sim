"""Verify an original single-launch CA window after the optional reporting job hit its disk cap."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.rms_ordered_verify.verify import checked, digest, events, tables, text_bytes
from development.validation.sku19.campaign import write
from hashlib import sha256


def verify(root, original, *, capture_folder=None):
    original_data = json.loads(original.read_text())
    group = original.parent.parent.name
    chip = original_data['chip']
    t, n, d = original_data['shape']
    post = group == 'dispatch'
    api = 'aclnnMhcPost' if post else 'aclnnMhcPreSinkhorn'
    folder = capture_folder or root/'runs'/(group+'21')/original.stem
    evidence_root = ROOT if capture_folder else root
    log = (folder/'run.log').read_text()
    profile, = folder.glob('profile/OPPROF_*')
    captures = list(profile.glob('device0/*/0/simulator/trace.json'))
    trace = captures[0] if captures else profile/'simulator/trace.json'
    name, = re.findall(r'Start profiling on kernel: (\S+)', log)
    checked(re.findall(r'^runtime_soc=(\S+)', log, re.M) == [chip], 'wrong runtime chip')
    checked(('lpddr5_6400M_bailu.ini' if '950PR' in chip else 'hbm3_8000M.ini') in log, 'memory model')
    checked(f'api={api} T={t} N={n} D={d} dtype=BF16' in log, 'host tensors')
    checked(all(mark in log for mark in ('SMOKE_PASS','All threads are joined.','Model stopped successfully.','[TASK_DONE]')), 'device termination')
    checked(not re.search(r'SMOKE_FAIL|\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)', log, re.I), 'device error')
    checks = re.findall(r'^correctness [^\r\n]+', log, re.M)
    sizes = [(k,int(v)) for k,v in re.findall(r'^correctness name=(\w+) count=(\d+) bad=0\b', log, re.M)]
    checked(sizes == ([('output',t*n*d)] if post else [('hin',t*d),('hpost',t*n),('hres',t*n*n)]) and len(checks)==len(sizes), 'numeric coverage')
    elf = profile/'dump/aicore_binary.o'
    base_name = re.match(r'^(?:MhcPost|MhcPreSinkhorn)_[a-f0-9]{32}', name).group()
    library = root/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer'
    control, = library.glob('*/'+base_name+'.o')
    checked(text_bytes(elf) == text_bytes(control), 'ELF text differs from installed library')
    duration, = tables(log)
    bounds = {}; ends = Counter(); queues = Counter(); mmads = 0
    for event in events(trace):
        core,name_i = event['pid'],event['name']
        checked(re.fullmatch(r'core\d+\.(?:vec|cube)core\d+',core), 'unexpected core')
        start = round(event['ts']*1650); end = round((event['ts']+event['dur'])*1650)
        checked(end >= start, 'negative instruction time')
        if core not in bounds:
            bounds[core] = [start, end]
        else:
            bounds[core][0] = min(bounds[core][0],start); bounds[core][1] = max(bounds[core][1],end)
        if name_i == 'END': ends[core] += 1
        mmads += name_i == 'MMAD'; queues[event['tid']] += 1
    checked(bounds.keys()==duration.keys(),f'core roster differs: trace {len(bounds)} official {len(duration)}')
    for core,(a,b) in bounds.items():
        checked(abs((b-a)/1650 - float(duration[core][0])) < .015, 'official rounded duration '+core)
    missing=[c for c in bounds if ends[c]!=1]
    raw_proof=None
    if missing and not post and not capture_folder:
        from development.validation.sku22.mhc_raw_window import repair
        bounds,raw_proof=repair(root,profile,name,duration,bounds,text_bytes(control),ROOT/'results/v0.22/recovered',ends)
    else:
        checked(not missing,f'missing/duplicate physical END on {missing}')
    start = min(a for a,b in bounds.values()); finish = max(b for a,b in bounds.values())
    return dict(kernel=name,api=api,ca_us=(finish-start)/1650,active_aic=sum('cube' in c for c in bounds),
                active_aiv=sum('vec' in c for c in bounds),correctness=checks,mmad=mmads,queue_events=dict(queues),
                first_tick=start,last_tick=finish,cycles_per_us=1650,physical_end_cores=len(bounds),windows_ticks=bounds,
                timing_source='physical raw popped/executed per core' if raw_proof else 'exported complete all-core trace',
                raw_proof=raw_proof,queue_counts_scope='exported instructions only' if raw_proof else 'all-core complete',
                trace=str(trace.relative_to(evidence_root)),trace_sha256=digest(trace),binary_sha256=digest(elf),
                control_elf=str(control.relative_to(root)),elf_text_sha256=sha256(text_bytes(elf)).hexdigest(),
                log_sha256=digest(folder/'run.log'), cost=json.loads((folder/'cost.json').read_text()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);args=parser.parse_args()
    root=args.archive.resolve(); out=ROOT/'results/v0.22/recovered';out.mkdir(parents=True,exist_ok=True)
    for group in ('fusion','dispatch'):
        for source in sorted((root/'results/v0.21'/group/'receipts').glob('mhc*.json')):
            data=json.loads(source.read_text());retry=source.parent.parent/'retries'/source.name
            if data['qualified'] or (retry.exists() and json.loads(retry.read_text())['qualified']):continue
            dest=out/source.name
            if dest.exists() and json.loads(dest.read_text())['qualified']:continue
            receipt=dict(shape=data['shape'],chip=data['chip'],op='mhc_post' if group=='dispatch' else 'mhc_pre_sinkhorn',
                         original=str(source.relative_to(root)),original_sha256=digest(source),new_ca_seconds=0)
            try:
                receipt['observation']=verify(root,source); receipt['qualified']=True
            except (OSError,ValueError,IndexError,StopIteration) as exc:
                receipt.update(qualified=False,reason=str(exc) or repr(exc))
            # A registered clean retry may complete while the read-only archive is being parsed.
            if dest.exists() and json.loads(dest.read_text())['qualified']:
                continue
            write(dest,receipt);print(dest.stem,receipt['qualified'],receipt.get('reason',''),flush=True)

if __name__=='__main__':main()
