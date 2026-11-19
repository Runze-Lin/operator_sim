"""Recover an interrupted CA supervisor without replaying the simulated kernel."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def raw_bytes(folder):
    total=0
    for base,_,files in os.walk(folder):
        for name in files:
            try:total+=(Path(base)/name).stat().st_size
            except FileNotFoundError:pass
    return total


def timer(folder):
    """Only bind an existing /usr/bin/time process owned by this account in the
    exact case directory and a private process group created by the collector."""
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid() or (proc/'comm').read_text().strip()!='time':continue
            pid=int(proc.name)
            if Path(os.readlink(proc/'cwd'))==folder and os.getpgid(pid)==pid:
                return pid
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return None


def recover_cost(folder,wall_s,cap_bytes):
    """Return original-process cost (no rerun). Peak disk is a lower bound
    because the departed supervisor's earlier samples were not persisted."""
    folder=Path(folder).resolve()
    logfile=folder/'run.log';cpu_file=folder/'cpu.txt'
    born=subprocess.check_output(['stat','-c','%w',str(logfile)],text=True).strip()
    if born=='-':raise RuntimeError('The run log filesystem does not report creation time')
    date_part,fraction_zone=born.split('.',1)
    fraction,zone=fraction_zone.split()
    started=datetime.strptime(f'{date_part}.{fraction[:6]} {zone}',
                              '%Y-%m-%d %H:%M:%S.%f %z').astimezone(timezone.utc)
    begin_watch=datetime.now(timezone.utc).isoformat();peak=0;stopped=None
    pid=timer(folder)
    if pid is not None:
        while True:
            try:alive=Path(os.readlink(f'/proc/{pid}/cwd'))==folder and os.getpgid(pid)==pid
            except (FileNotFoundError,ProcessLookupError):alive=False
            if not alive:break
            peak=max(peak,raw_bytes(folder))
            if time.time()-started.timestamp()>wall_s:stopped='original wall cap'
            if peak>cap_bytes:stopped='original disk cap'
            if stopped:
                # A new session plus validated account, executable and cwd
                # identify only the simulated process group for this case.
                if timer(folder)==pid:os.killpg(pid,signal.SIGTERM)
                time.sleep(3)
                if timer(folder)==pid:os.killpg(pid,signal.SIGKILL)
                break
            time.sleep(4)
    cpu=cpu_file.read_text().strip().splitlines() if cpu_file.exists() else []
    success_shape=(len(cpu)==1 and re.fullmatch(r'[\d.]+ [\d.]+',cpu[0]))
    code=0 if success_shape and not stopped else None
    if not success_shape and cpu:
        code_matches=re.search(r'exited with non-zero status (\d+)',cpu[0])
        sig_matches=re.search(r'terminated by signal (\d+)',cpu[0])
        if code_matches:code=int(code_matches[1])
        if sig_matches:code=-int(sig_matches[1])
    if not cpu and not stopped:
        raise RuntimeError(f'GNU time has no completion record and no owned timer remains in {folder}')
    end_ts=cpu_file.stat().st_mtime if cpu else time.time()
    finished=datetime.fromtimestamp(end_ts,timezone.utc);raw=raw_bytes(folder)
    return dict(started_utc=started.isoformat(),finished_utc=finished.isoformat(),seconds=(finished-started).total_seconds(),
                raw_bytes=raw,peak_bytes=max(peak,raw),peak_is_lower_bound=True,
                cpu_seconds=sum(map(float,cpu[-1].split())) if cpu and re.fullmatch(r'[\d.]+ [\d.]+',cpu[-1]) else None,
                exit_code=code,stop_reason=stopped,recovered=True,began_recovery_utc=begin_watch,
                recovery_evidence='run.log birth and cpu.txt final write delimit wall; original GNU time success/exit/signal record; retained bytes and samples after supervisor recovery bound peak disk from below.')
