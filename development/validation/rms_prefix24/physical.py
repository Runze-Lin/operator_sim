"""Qualify the physical window without requiring exporter-generated END_LABELs."""
from collections import Counter, defaultdict
from campaign import verify


def measure(folder, chip, job, table, elf):
    shape = job['shape']
    binary = folder/'dump/aicore_binary.o'
    verify.checked(verify.text_bytes(binary)==elf,'ELF text mismatch')
    base = int((folder/'dump/pc_start_addr.txt').read_text(),16)
    cores = {}
    writes = Counter()
    row_count = 0
    for shard in sorted((folder/'simulator').glob('core*/trace.json')):
        core = shard.parent.name
        span = verify.Span()
        for e in verify.events(shard):
            if e['name']!='END_LABEL':
                span.add(e,base)
        verify.checked(span.entry==[span.first] and len(span.end)==1 and span.end[0][2]==0x51d0,'absent kernel entry/END on '+core)
        verify.checked(span.widths=={2*shape[1]} and set(span.writes)=={0x4f38,0x5128} and span.writes[0x4f38]==4*span.yrows and span.writes[0x5128]==2*shape[1]*span.yrows,'invalid output writes on '+core)
        verify.checked(core in table and abs((span.last-span.first)/verify.CLOCK-float(table[core][0]))<=.01,'physical window and official per-core length disagree')
        cores[core] = span
        writes.update(span.writes)
        row_count += span.yrows
    verify.checked(set(cores)==set(table) and len(cores)==job['active_aiv'],'physical core roster')
    verify.checked(row_count==shape[0] and writes=={0x4f38:4*shape[0],0x5128:2*shape[0]*shape[1]},'whole tensor output writes')
    path = folder/'simulator/trace.json'
    aggregate = defaultdict(verify.Span)
    for e in verify.events(path):
        if e['name']!='END_LABEL':
            aggregate[e['pid']].add(e,base)
    verify.checked(cores.keys()==aggregate.keys() and all(s.signature()==aggregate[k].signature() for k,s in cores.items()),'per-core versus aggregate physical instruction mismatch')
    first = min(s.first for s in cores.values()); last = max(s.last for s in cores.values())
    return dict(ca_us=(last-first)/verify.CLOCK, trace=verify.rel(path),sha256=verify.digest(path),kernel=verify.KERNEL,
                binary_sha256=verify.digest(binary),first_tick=first,last_tick=last)
