"""Estimate FP4 service prices from eight previous windows' physical core stages."""
import json
from math import ceil, log2
from statistics import median


def fit(rows):
    # One vote per shape/SKU for ordinary response, control and vector entry.
    per_chip={}
    scale_uncontended=[];ordinary_gaps=[];partial_gaps=[];small_read=[];fill=[]
    for suffix,launch,spread,readlat in (('dt',4686.5,219.92134595794607,1029.7401547091642),
                                         ('pr',4253.,227.76564103929616,954.1200773545821)):
        fronts=[];spans=[];normal_read=[];vector=[]
        for name,cores in rows.items():
            if not name.endswith(suffix):
                continue
            plain=[c for c in cores if c['zero_vf_markers']<100]
            odd=[c for c in cores if c not in plain]
            issue=[c['first_read'] for c in plain]
            fronts.append(min(issue))
            spans.append((max(issue)-min(issue))/log2(len(cores)))
            if odd:
                fill.append(median(c['first_read'] for c in odd)-median(issue))
                small_read.extend(c['read_latency']-c['read_bytes']/48.362955988428816 for c in odd)
            normal_read.append(median(c['read_latency']-c['read_bytes']/48.362955988428816 for c in plain))
            vector.append(median(c['write_issue']-c['read_retire']-
                                 (15*ceil(c['read_bytes']/512)+16*ceil(c['read_bytes']/8192)) for c in plain))
            bulk=[c for c in plain if c['writers']<=2]
            if len(bulk)==len(plain):
                scale_uncontended.append(median(c['scale_cycles']-c['scale_bytes']/92.02711547224736 for c in bulk))
            grains={}
            for c in cores:
                grains.setdefault(c['scale_addr']//32,[]).append(c['scale_retire'])
            # There are two small-fragment geometries in the archived corpus.
            # Bulk windows' strided multirow descriptors cannot yield an II
            # from the first-address grain's retirements.
            if all(c['scale_bytes']<=8 for c in cores):
                gaps=partial_gaps if odd else ordinary_gaps
                for times in grains.values():
                    times.sort()
                    gaps.extend(b-a for a,b in zip(times,times[1:]))
        per_chip[suffix]=dict(first_issue=median(fronts)/launch,arrival=median(spans)/spread,
                              read=median(normal_read)/readlat,read_cycles=median(normal_read),
                              vector=median(vector))
    avg=lambda key:median(c[key] for c in per_chip.values())
    return dict(chip_ratios=per_chip,first_issue_ratio=avg('first_issue'),arrival_ratio=avg('arrival'),
                read_ratio=avg('read'),short_read_ratio=median(small_read)/avg('read_cycles'),
                vector_setup_cycles=avg('vector'),full_buffer_clear_cycles=median(fill),
                scale_base_cycles=median(scale_uncontended),ordinary_grain_ii=median(ordinary_gaps),
                partial_grain_ii=median(partial_gaps))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rows',type=Path)
    args=parser.parse_args()
    print(json.dumps(fit(json.loads(args.rows.read_bytes())),indent=2))
