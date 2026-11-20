"""Sample the authorized CA job's scheduler counters without perturbing its CPU mask."""
import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import time


def snapshot(tids):
    records = {}
    for tid in tids:
        try:
            fields = dict(line.split(':', 1) for line in (tid / 'status').read_text().splitlines() if ':' in line)
            runtime_ns, waiting_ns, switches = map(int, (tid / 'schedstat').read_text().split())
            stat = (tid / 'stat').read_text().rsplit(')', 1)[1].split()
            records[tid.name] = {
                'mask': fields['Cpus_allowed_list'].strip(),
                'state': stat[0], 'last_cpu': int(stat[36]), 'runtime_ns': runtime_ns,
                'waiting_ns': waiting_ns, 'switches': switches,
                'wchan': (tid / 'wchan').read_text().strip(),
            }
        except FileNotFoundError:
            continue
    return records


def counter(records, field):
    return dict(Counter(record[field] for record in records))


def cpus_used():
    usage={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if not fields[0].startswith('cpu') or not fields[0][3:].isdigit():
            continue
        times=list(map(int,fields[1:9]))
        usage[int(fields[0][3:])]=(sum(times),times[3]+times[4],times[1])
    return usage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pid', type=int)
    p.add_argument('destination', type=Path)
    p.add_argument('--seconds', type=float, default=4)
    args = p.parse_args()
    proc = Path('/proc') / str(args.pid)
    root = Path(__file__).resolve().parents[3]
    if (proc / 'exe').resolve() != root / 'build/vector-v8/MHC' or proc.stat().st_uid != os.getuid():
        raise ValueError('not this project’s authorized Post CA process')
    tids = list((proc / 'task').iterdir())
    start = snapshot(tids); host_start=cpus_used()
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.monotonic()
    time.sleep(args.seconds)
    end = snapshot(tids); host_end=cpus_used()
    dt = time.monotonic() - t0
    common = end.keys() & start.keys()
    groups = {}
    for mask in sorted({end[t]['mask'] for t in common}):
        group = [t for t in common if end[t]['mask']==mask]
        delta = {k: sum(max(0, end[t][k] - start[t][k]) for t in group) for k in ('runtime_ns','waiting_ns','switches')}
        groups[mask] = {
            'n_threads':len(group), 'cpu_cores_used':delta['runtime_ns']/1e9/dt,
            'runnable_wait_thread_s':delta['waiting_ns']/1e9, 'scheduler_switches':delta['switches'],
            'states':counter((end[t] for t in group),'state'), 'wchan':counter((end[t] for t in group),'wchan'),
        }
    capacity={}
    for cpu in sorted(os.sched_getaffinity(args.pid)):
        ticks, idle, nice = (n-p for n,p in zip(host_end[cpu],host_start[cpu]))
        capacity[cpu]=dict(idle_fraction=idle/max(1,ticks),niced_user_fraction=nice/max(1,ticks))
    out = dict(pid=args.pid,utc=utc,elapsed_s=dt,host_logical_cpus=os.cpu_count(),loadavg=list(os.getloadavg()),mask_cpu_host_usage=capacity,
               n_threads=len(common),last_cpu=counter(end.values(),'last_cpu'),by_affinity=groups,
               task='read-only Linux schedstat snapshot; runnable time, queue wait and affinity only. No userspace stack or spin-loop attribution.')
    dest=args.destination
    with dest.open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps({'utc':utc,'sampled':len(common),'affinity_groups':len(groups),
                      'executed_cores':sum(g['cpu_cores_used'] for g in groups.values())}))


if __name__=='__main__': main()
