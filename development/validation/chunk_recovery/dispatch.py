"""Cross-check official QKV/gate/beta dispatch against wrapper entry addresses."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/chunk-recovery'
TAG = 'generic-fp16-c128-3x16'
KERNEL = ROOT / 'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = KERNEL / 'config/ascend950/ops_transformer/chunk_kda_fwd.json'
    entries = json.loads(config.read_text())['binList']
    contract = ['float16', 'float16', 'float16', 'float32', 'float32']
    chosen, = [item for item in entries if [x['dtype'] for x in item['inputs'][:5]] == contract]
    metadata = KERNEL / 'ascend950/ops_transformer/chunk_kda_fwd' / Path(chosen['binInfo']['jsonFilePath']).name
    obj = metadata.with_suffix('.o')
    key, = [k for k in json.loads(metadata.read_text())['kernelList'] if k['tilingKey'] == 1]
    syms = subprocess.check_output(['readelf','--wide','-s',str(obj)],text=True)
    offsets = {}
    for pipe in ('aic','aiv'):
        name = key['kernelName']+'_mix_'+pipe
        m = re.search(r'^\s*\d+:\s+([0-9a-f]+)\s+\S+\s+FUNC\s+.*\s'+name+r'$', syms,re.M)
        offsets[pipe] = int(m[1],16)
    trace, = (ROOT/'runs/chunk-recovery'/TAG).glob('profile/*/record/log_ca/core0.wrapper_log.dump')
    raw = trace.read_text()
    starts = {int(core):int(pc,16) for core,pc in re.findall(r'Core 0 subcore (\d+) start pc (0x[0-9a-f]+)',raw)}
    if len(starts)!=3 or starts[1]!=starts[2] or starts[1]-starts[0] != offsets['aiv']-offsets['aic']:
        raise ValueError('runtime start addresses inconsistent with expected installed MIX')
    log = (ROOT/'runs/chunk-recovery'/TAG/'run.log').read_text()
    faults = re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)', log)
    fault_offsets = {p:int(p,16)-starts[0]+offsets['aic'] for p in dict.fromkeys(faults)}
    aic_faults = {p:off for p,off in fault_offsets.items() if offsets['aic'] <= off < offsets['aiv']}
    decoded = {}
    if aic_faults:
        dis = subprocess.check_output(['bash',str(ROOT/'scripts/cann-run.sh'),'llvm-objdump','-d',
            '--start-address='+hex(min(aic_faults.values())), '--stop-address='+hex(max(aic_faults.values())+4), str(obj)],text=True)
        codes = {int(off,16):op for off,op in re.findall(r'^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)',dis,re.M)}
        decoded = {p:{'offset':hex(off), 'opcode':codes[off]} for p,off in aic_faults.items()}
    result = dict(inputs=contract, fault_messages_total=len(faults), distinct_fault_pcs=len(set(faults)),metadata_match=key,metadata_sha256=digest(metadata),package_elf=obj.name,package_elf_sha256=digest(obj),
                  expected_symbol_offsets={k:hex(v) for k,v in offsets.items()}, wrapper_subcore_entry_pcs={k:hex(v) for k,v in starts.items()},
                  faults_in_aic_segment=decoded, expected_numerical_checks={"output":48,"aqk":384,"akk":384}, entry_delta_bytes=starts[1]-starts[0], config_sha256=digest(config),wrapper_path=str(trace.relative_to(ROOT)),
                  scope='Observed one AIC and both Vector subcores at the addresses expected by the FP16 generic family. Exact file selection is inferred from dispatch metadata; entry addresses do not distinguish all same-family blobs.')
    (OUT/f'{TAG}-dispatch.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
