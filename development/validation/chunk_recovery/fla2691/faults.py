"""Map every repeated hardware SET to its instruction in the captured wheel executable."""
from collections import Counter
from pathlib import Path
import json
import re
import subprocess

from probe import OUT, RAW, ROOT, CAPTURE, digest, put, rel


def main():
    [profile]=list(CAPTURE.glob('profile/OPPROF_*'))
    dirs=sorted(profile.glob('**/ChunkKdaFwd_*/0/dump/aicore_binary.o'))
    if not dirs:
        raise ValueError('no captured Chunk MIX binary')
    binary=dirs[0]
    base=int((binary.parent/'pc_start_addr.txt').read_text(),16)
    log=(CAPTURE/'run.log').read_text(errors='replace')
    occurs=Counter(int(s,16)-base for s in re.findall(r'already has same set_flag! pc:(0x[0-9a-f]+)',log))
    if not occurs:
        raise ValueError('no repeated SET to decode')
    # Disassemble exactly the PC ranges needed in one compiler tool invocation.
    dis=subprocess.run(['bash',str(ROOT/'scripts/cann-run.sh'),'llvm-objdump','-d',
                         '--start-address='+hex(min(occurs)), '--stop-address='+hex(max(occurs)+8),str(binary)],
                        text=True,capture_output=True,check=True).stdout
    words={}
    for line in dis.splitlines():
        m=re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)(.*)$',line)
        if m and int(m[1],16) in occurs:
            words[int(m[1],16)]=(m[2],m[3],m[4].strip())
    if words.keys()!=occurs.keys():
        raise ValueError('unresolved PCs')
    findings=json.loads((RAW/'sync_audit.txt').read_text())['findings']
    graph=json.loads((RAW/'ascendc_flow_analyzer.txt').read_text())
    res=dict(base=hex(base),captured_binary=rel(binary),captured_binary_sha256=digest(binary),
             repeats=sum(occurs.values()),distinct=len(occurs),
             instructions=[dict(offset=hex(pc),raw=words[pc][0],opcode=words[pc][1],operands=words[pc][2],repeats=n) for pc,n in occurs.items()],
             opcodes=dict(Counter(words[pc][1] for pc in occurs)),static_finding_counts=dict(Counter(f['code']+' '+f['severity'] for f in findings)),
             static_high_findings=[f for f in findings if f['severity'] in ('红线','高')],
             static_graph_findings=graph['findings'],
             scope='Static analyzer candidates are cross-branch syntactic findings; runtime counts above are vendor CA errors matched to ELF PCs. A full CA window plus goldens is required separately.')
    put(OUT/'faults.json',res)
    print(res['repeats'],res['opcodes'])

if __name__=='__main__':
    main()
