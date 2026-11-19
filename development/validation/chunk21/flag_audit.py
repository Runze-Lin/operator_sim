"""Find the preceding matching flag instructions on the failing core's raw retirement stream."""
from collections import defaultdict
import subprocess
import sqlite3
import struct
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
RECORD=ROOT/'runs/sku17/extension/kda_chunk_1_64_2_2_128_128_Ascend950PR_957b'
pattern=re.compile(r'^\[info\] \[(\d+)\] \(PC: (0x[\da-f]+)\).*? (SET_FLAG|WAIT_FLAG)\s+(\{.+\})')
events=defaultdict(list)
faults=set(re.findall(r'already has same set_flag! pc:(0x[\da-f]+)',(RECORD/'run.log').read_text()))
dump,=RECORD.glob('profile/*/device0/ChunkKdaFwd*/0/dump')
base=int((dump/'pc_start_addr.txt').read_text(),16)
path=dump/'core0.cubecore0.instr_log.dump'
for line in path.open():
    match=pattern.match(line)
    if not match:
        continue
    cycle,pc,op,encoded=match.groups()
    payload=json.loads(encoded)
    # CA describes the waiting pipe in target_pipe on WAIT; normalize direction.
    target=payload['target_pipe']
    consumer=payload['consumer_pipe']
    flag=payload['flag_id']
    entry={'cycle':int(cycle),'op':op,'pc':pc,'offset':hex(int(pc,16)-base),
           'target_pipe':target,'consumer_pipe':consumer,'flag_id':flag,'simulator_error':pc in faults}
    events[flag].append(entry)
report=[]
for flag,history in sorted(events.items()):
    for i,current in enumerate(history):
        if not current['simulator_error']: continue
        prior=[x for x in history[:i] if {x['target_pipe'],x['consumer_pipe']}=={current['target_pipe'],current['consumer_pipe']}]
        report.append({'flag_id':flag,'previous_same_pipe_pair':prior[-1] if prior else None,'flagged':current})
if len(report)!=3:
    raise ValueError('Unexpected number of matching simulator errors')
evidence={'shape':[1,64,2,2,128,128], 'sku':'Ascend950PR_957b','role':'v0.17 diagnostic partial, excluded from timing',
          'kernel':dump.parents[1].name,'core':'core0.cubecore0','code_load_base':hex(base),
          'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,dump/'aicore_binary.o',RECORD/'run.log')},
          'pairs':report,'scope':'Same core, unordered pipe pair, same flag ID. Each consecutive matching pair is SET/SET. Source line unresolved; ELF lacks debug lines.'}
# Same host executable/contract with DT's compact official simulator capture.
dt=ROOT/'runs/v8cube-isolated/kda_chunk_safe_64_2'
record,=dt.glob('profile/*/record/0001_*.db')
with sqlite3.connect(record.resolve().as_uri()+'?mode=ro', uri=True) as db:
    n=db.execute('SELECT count(*) FROM ExecutedInstructions').fetchone()[0]
dtlog=(dt/'run.log').read_text()
dt_errors=set(re.findall(r'already has same set_flag! pc:(0x[\da-f]+)',dtlog))
package=(ROOT/'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/chunk_kda_fwd')
def text_digest(obj):
    data=obj.read_bytes()
    shoff=struct.unpack_from('<Q',data,40)[0]
    shsize,num,names=struct.unpack_from('<HHH',data,58)
    start,size=struct.unpack_from('<QQ',data,shoff+names*shsize+24)
    strings=data[start:start+size]
    for i in range(num):
        idx=shoff+i*shsize
        pos=struct.unpack_from('<I',data,idx)[0]
        if strings[pos:].split(b'\0',1)[0] == b'.text':
            start,size=struct.unpack_from('<QQ',data,idx+24)
            return hashlib.sha256(data[start:start+size]).hexdigest()
    raise ValueError('missing ELF text section')
evidence['dt_same_contract']={'runtime_soc':re.search(r'runtime_soc=(\w+)',dtlog)[1],
    'gate_dtype':'fp32','lower_bound':-0.03125,'captured_main_instruction_rows':n,
    'errors_pc':sorted(dt_errors),'same_pc_set_as_pr':dt_errors==faults,
    'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (record,dt/'run.log')}}
evidence['installed_elf_text']={path.name:text_digest(path) for path in package.glob('ChunkKdaFwd_*.o')}
evidence['captured_pr_elf_text']=text_digest(dump/'aicore_binary.o')
evidence['elf_decoding'] = {}
for pair in report:
    for event in (pair['previous_same_pipe_pair'], pair['flagged']):
        off = int(event['offset'], 16)
        listing = subprocess.check_output(
            ['bash', str(ROOT/'scripts/cann-run.sh'), 'llvm-objdump', '-d',
             '--start-address='+hex(off), '--stop-address='+hex(off+4), str(dump/'aicore_binary.o')],
            text=True)
        parsed = re.search(rf'^\s*{off:x}:\s+([a-f0-9]+)\s+(\S+)\s+(X\d+)', listing, re.M)
        if not parsed or parsed[2] != 'SET_FLAG.M.MTE1':
            raise ValueError(f'missing expected instruction at {off:x}')
        evidence['elf_decoding'][hex(off)] = {'machine_word':parsed[1], 'instruction':parsed[2],
                                              'event_register':parsed[3]}
(ROOT/'results/v0.21/chunk/pr-flag-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
