"""Summarize simulated memory-channel counters while vendor text logs still exist."""
from pathlib import Path
import re
import time
from large_cast import OUT, RAW, read, write

PERIOD = re.compile(rb'Current time: (\d+), tDFI: ([\d.]+) ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
COMMANDS = re.compile(rb'^Read\s+: (\d+)\s+\| Total reads\s+: (\d+)\s+\| Write\s+: (\d+)\s+\| Total writes\s+: (\d+)',re.M)


def counters(folder):
    files=sorted((folder/'log').glob('lpddr_state*.log'))
    if not files:return None
    dmc=ddr=r=w=panels=0
    last_times=set()
    for file in files:
        # Each file covers one memory controller; bytes in the header are per
        # period, while the final cumulative commands count spans all periods.
        sub_r=sub_w=read_cmd=write_cmd=0
        end_time=None
        with file.open('rb') as f:
            for line in f:
                m=PERIOD.search(line)
                if m:
                    assert float(m[2])==1
                    end_time=int(m[1]);dmc+=int(m[3]);ddr+=int(m[4]);panels+=1
                m=COMMANDS.match(line)
                if m:
                    sub_r+=int(m[1]);sub_w+=int(m[3]);read_cmd=int(m[2]);write_cmd=int(m[4])
        if end_time is None:return None
        assert (sub_r,sub_w)==(read_cmd,write_cmd)
        r+=read_cmd;w+=write_cmd;last_times.add(end_time)
    if len(files)!=128 or len(last_times)!=1:return None
    return dict(channel_files=len(files),complete_channel_reports=panels,last_controller_time_ns=last_times.pop(),
                dmc_bytes=dmc,ddr_bytes=ddr,read_commands=r,write_commands=w,
                scope='all 128 DT memory channels from CA process initialization to its final reporting interval; includes every allocated channel, all metadata and any background traffic')


def archive_native(t,h):
    from native import NATIVE
    name=f'cast_{t}_{h}'
    file=OUT/(name+'_native.json')
    out=OUT/(name+'_native-dram.json')
    if not file.exists() or out.exists():return None
    result=read(file)
    if not result['qualified_core_envelope']:return None
    counts=counters(NATIVE/name)
    if counts:
        counts.update(shape=[t,h],logical_read_bytes=2*t*h,logical_write_bytes=4*t*h)
        write(out,counts)
    return counts


def save_official_counters():
    """The profiler unlinks controller logs on exit. Hard links retain the same
    inode (no file copy) while it is open. Keep the trace collector untouched."""
    import os
    from large_cast import DT, ROOT
    jobs=read(OUT/'registration.json')['jobs'][1:]
    while True:
        pending=False
        for j in jobs:
            name=j['id']+'_'+DT
            dest=ROOT/'runs/prefill20-memory'/name/'log'
            outcome=OUT/(name+'.json');summary=OUT/(name+'-official-dram.json')
            if summary.exists():continue
            for f in (RAW/name/'log').glob('lpddr_state*.log'):
                dest.mkdir(parents=True,exist_ok=True)
                try:os.link(f,dest/f.name)
                except (FileExistsError,FileNotFoundError):pass
            if not outcome.exists():
                pending=True
                continue
            if read(outcome)['qualified']:
                counts=counters(dest.parent)
                if counts:
                    t,h=j['shape']
                    counts.update(shape=j['shape'],logical_read_bytes=2*t*h,logical_write_bytes=4*t*h)
                    write(summary,counts)
        if not pending:return
        time.sleep(20)


if __name__=='__main__':
    save_official_counters()
