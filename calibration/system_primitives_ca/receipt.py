"""Qualify synchronized primitive calls against numeric goldens, profiler rosters and installed objects."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from calibration.__main__ import require
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows

SEQUENCES = {
    "mask": ("LtScalar","GeScalar","BitwiseOrTensor","Subs","InplaceMaskedFillScalar"),
    "math": ("Neg","PowTensorScalar","ReduceSum","Mean","Rsqrt"),
    "memory": ("Embedding","GatherV2","Cast","Cat","ConstantPadNd"),
}


def verify(directory,opp):
    raw = (directory/"run.log").read_text()
    cost = json.loads((directory/"cost.json").read_text())
    sku,group,rows=re.search(r"^runtime_soc=(\S+) group=(\w+) rows=(\d+)$",raw,re.M).groups()
    require(cost["exit_code"]==0 and "SMOKE_PASS" in raw and "All task success." in raw
            and "Profiling data parse finished." in raw and MEMORY_CONFIG[sku] in raw,"Incomplete run")
    launches = re.findall(r"^launch=(\w+)$",raw,re.M)
    checks = re.findall(r"^verified (\w+) n=(\d+) bad=(\d+)$",raw,re.M)
    kernels = re.findall(r"Start profiling on kernel: (\S+)",raw)
    tables = re.split(r"^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$",raw,flags=re.M)[1:]
    require(launches==["aclnn"+n for n in SEQUENCES[group]]
            and [api for api,n,bad in checks if int(n)>0 and int(bad)==0]==launches,"Golden or ABI failed")
    require(len(tables)==len(kernels)==len(launches),"Ambiguous launch roster")
    base,= (directory/"profile").glob("OPPROF_*")
    seen=Counter(); stages=[]
    for api,kernel,table,(_,n,_) in zip(launches,kernels,tables,checks):
        d=base/kernel/str(seen[kernel]);seen[kernel]+=1
        elf=sha(d/"dump/aicore_binary.o")
        filename=re.sub(r"_\d+(?:_mix_ai[cv])?$","",kernel)+".o"
        matches=[p for p in opp.glob("*/*/"+filename) if sha(p)==elf]
        require(len(matches)==1,f"Cannot identify {kernel}")
        entries=CORE.findall(table);roster={k:float(v) for k,v in entries}
        require(roster and len(roster)==len(entries),"Missing or duplicate core")
        cores, hashes=core_windows(d/"simulator",roster)
        ticks=max(c["retire_tick"] for c in cores.values())-min(c["first_tick"] for c in cores.values())
        stages.append(dict(api=api,correct_elements=int(n),kernel=kernel,object=str(matches[0].relative_to(opp)),
                           cycles=ticks,window_us=round(ticks/1650,6),aiv=len(roster),elf_sha256=elf,
                           trace_sha256=sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()))
    return dict(chip=sku,group=group,rows=int(rows),stages=stages,wall_s=round(cost["seconds"],2),
                memory_model=MEMORY_CONFIG[sku],probe_sha256=cost["probe_sha256"],log_sha256=sha(directory/"run.log"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",nargs="+",type=Path);p.add_argument("--opp",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    records=sorted((verify(d,a.opp) for d in a.raw),key=lambda row:(row["group"],row["chip"],row["rows"]))
    a.out.write_text(json.dumps(records,indent=2)+"\n")
