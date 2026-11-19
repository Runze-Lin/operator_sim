"""Pair Cast's two-slot service budgets with the original terminal-core trace."""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.22/cast"
SHAPE = [511, 5120]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"development/validation/sku17"))
from operator_model import estimate, load_chip
from campaign_collect import events


def read(path):
    return json.loads(path.read_text())


def measured(raw_root, row):
    stage = row["stage"]
    core = max(stage["windows"],key=lambda c:stage["windows"][c][1])
    shard = raw_root / Path(stage["trace"]).parent / core / "trace.json"
    first, last = float("inf"), -float("inf")
    dma = {"MTE2":[], "MTE3":[]}
    begins, conv = [], []
    for e in events(shard):
        if e["ph"] != "X":
            continue
        a,b = e["ts"],e["ts"]+e["dur"]
        first,last = min(first,a),max(last,b)
        if e["name"] == "MOV_SRC_TO_DST_ALIGNv2":
            d = json.loads(e["args"]["detail"])
            q = d["instr_type"]
            if q not in dma or d["is_duration_excluded"]:
                continue
            is_read = q=="MTE2"
            assert (d["src_mem"],d["dst_mem"]) == (("OUT","UB") if is_read else ("UB","OUT"))
            gm,ub = (d["src_addr"],d["dst_addr"]) if is_read else (d["dst_addr"],d["src_addr"])
            dma[q].append(dict(bytes=d["processed_bytes"],gm=gm,ub=ub,window=[a,b]))
        elif e["name"]=="RV_VAG" and e["args"]["pc_addr"]=="0x9000d0d900":
            begins.append(a)
        elif e["name"]=="RV_VCVT_F2F" and e["dur"]>0:
            conv.append((a,b))
    lo,hi = stage["windows"][core]
    assert abs((last-first)-(hi-lo)) < .0001 and hi==row["ca_us"]
    with shard.open("rb") as source:
        digest=hashlib.sha256()
        for block in iter(lambda:source.read(1024**2),b""):
            digest.update(block)
    origin=first-lo
    begins.sort()
    tiles=[dict(slot=i%2) for i in range(len(begins))]
    for q,f in dma.items():
        f.sort(key=lambda d:d["gm"])
        assert len(f)==len(tiles) and all(x["gm"]+x["bytes"]==y["gm"] for x,y in zip(f,f[1:]))
        assert all(x["ub"]==f[i%2]["ub"] for i,x in enumerate(f)) and f[0]["ub"]!=f[1]["ub"]
        for t,x in zip(tiles,f):
            t[q]={**x,"window":[a-origin for a in x["window"]]}
    groups=[[] for t in tiles]
    for a,b in conv:
        i=bisect_right(begins,a)-1
        assert i>=0
        groups[i].append((a,b))
    for t,ops in zip(tiles,groups):
        t["n"]=t["MTE2"]["bytes"]//2
        assert t["MTE2"]["bytes"]*2==t["MTE3"]["bytes"]
        assert len(ops)==math.ceil(t["n"]/64)
        t["vcvt"]=[min(a for a,b in ops)-origin,max(b for a,b in ops)-origin]
        assert t["MTE2"]["window"][1] <= t["vcvt"][0] <= t["vcvt"][1] <= t["MTE3"]["window"][0]
    return dict(core=core,begin=lo,total_us=hi,raw_time_origin=origin,tiles=tiles,
                active_aiv=stage["active_aiv"],active_aic=stage["active_aic"],
                shard=str(shard.relative_to(raw_root)),shard_sha256=digest.hexdigest(),
                all_core_trace_sha256=stage["trace_sha256"],binary_sha256=stage["binary_sha256"],
                kernel=stage["kernel"],api=row["api"],correctness=row["correctness"],wall_s=row["cost"]["seconds"])


def predicted(row, forecast):
    chip=load_chip(row["chip"])
    r=estimate("cast_bf16_fp32",SHAPE,chip=chip,dtype="bf16")
    assert math.isclose(r.latency_us,forecast["candidate_us"])
    d=r.projection["timing_diagnostics"];s=d["stages_us"];p=d["target_plan"]
    cap=p["ub_tile_elements"];n=p["busy_core_elements"]
    cycle=1000*chip.frequency_ghz
    first=[]
    end=0
    for k in ("arrival","control_to_read","read","vector","dependency_to_write","write"):
        first.append([k,end,end+s[k]]);end+=s[k]
    inp=d["input_slot_ii_cycles"]/cycle
    out=d["output_slot_ii_cycles"]/cycle
    tiles=[dict(n=min(cap,n),slot=0,first=first)]
    # The frozen route prices repeated elements as a fraction of a full
    # tile's throughput and residence. It doesn't locate later DMA issues.
    # Both chosen jobs are bound by the steady slot floor.
    for i in range(1,p["busy_core_waves"]):
        count=min(cap,n-i*cap);fraction=count/cap
        tiles.append(dict(n=count,slot=i%2,input_budget=[end,end+inp*fraction],
                          output_budget=[end,end+out*fraction]))
        end+=max(inp,out)*fraction
    assert math.isclose(end,r.latency_us,abs_tol=1e-9) and not s["shared_io"]
    return dict(plan=p,total_us=r.latency_us,old_us=forecast["baseline_us"],
                stages_us=s,slot_ii_us={"input":inp,"output":out},tiles=tiles)


def render(data,path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    ink,sub="#193342","#496574"
    pc={"MTE2":"#177d88","VEC":"#7455a4","MTE3":"#b34f2d"}
    model,ca=data["model"],data["ca"]
    nt=len(model["tiles"])
    fig,axes=plt.subplots(2,1,figsize=(11.3,6.9),sharex=True)
    fig.subplots_adjust(left=.20,right=.956,top=.775,bottom=.17,hspace=.51)
    fig.text(.20,.947,f'{data["sku_short"]} Cast  |  511 × 5,120',fontsize=22,weight="bold",color=ink)
    fig.text(.20,.904,f'BF16 → FP32   ·   {model["plan"]["active_aiv"]} AIVs   ·   {nt} tiles on the last core',fontsize=12,color=sub)
    handles=[Patch(color=pc["MTE2"],label="MTE2 / read"),Patch(color=pc["VEC"],label="Vector / convert"),
             Patch(color=pc["MTE3"],label="MTE3 / write"),Patch(facecolor="#c8e1e0",edgecolor=pc["MTE2"],
             hatch="///",label="Input slot bound"),Patch(facecolor="#efd7c9",edgecolor=pc["MTE3"],
             hatch="///",label="Output slot bound")]
    fig.legend(handles=handles,frameon=False,ncol=5,loc="upper left",bbox_to_anchor=(.19,.877),fontsize=9.2)
    def bar(ax,row,off,span,col,hatch=None):
        ax.barh(row+off,span[1]-span[0],left=span[0],height=.18,color=col,edgecolor=("#78949a" if hatch else col),
                lw=.45 if hatch else 0,hatch=hatch)
    for ax,src,title in [(axes[0],model,"Resource model  ·  first tile and steady slot bounds"),
                         (axes[1],ca,"CA trace  ·  "+ca["core"]+" (last to finish)")]:
        model_side=src is model
        for i,t in enumerate(src["tiles"]):
            y=nt-i-1
            if model_side and i==0:
                for phase,a,b in t["first"]:
                    pos,col={"arrival":(-.23,"#dce5e9"),"control_to_read":(-.23,"#bacbd1"),
                             "read":(.22,pc["MTE2"]),"vector":(0,pc["VEC"]),
                             "dependency_to_write":(0,"#d5dde2"),"write":(-.22,pc["MTE3"])}[phase]
                    bar(ax,y,pos,[a,b],col)
            elif model_side:
                bar(ax,y,.11,t["input_budget"],"#c8e1e0","///")
                bar(ax,y,-.11,t["output_budget"],"#efd7c9","///")
            else:
                for k,pos in [("MTE2",.22),("VEC",0),("MTE3",-.22)]:
                    bar(ax,y,pos,t["vcvt"] if k=="VEC" else t[k]["window"],pc[k])
            if i:ax.axhline(y+.5,color="#dfe6e9",lw=.7)
        end=src["total_us"]
        ax.axvline(end,color="#3c5662",ls=(0,(3,3)),lw=1.1)
        ax.set_title(title,loc="left",color=ink,weight="bold",fontsize=13)
        ax.set_title(f'{end:.3f} µs',loc="right",color=ink,weight="bold",fontsize=14)
        ax.set(ylim=(-.5,nt-.48),xlim=(0,max(ca["total_us"],model["total_us"])*1.035),
               yticks=[nt-i-1 for i in range(nt)],
               yticklabels=[f"Tile {i+1}  ·  slot {t['slot']}\n{t['n']:,} elements" for i,t in enumerate(src["tiles"])])
        ax.grid(axis="x",color="#dbe4e7",lw=.8);ax.set_axisbelow(True);ax.spines[:].set_visible(False)
        ax.tick_params(axis="both",length=0,labelcolor=ink)
    axes[1].xaxis.set_major_locator(MultipleLocator(.5))
    axes[1].set_xlabel("Device time from the first instruction across all AIVs (µs)",labelpad=9)
    fig.text(.20,.074,"Hatched bars show input and output reuse budgets after tile 1. Their lengths constrain throughput; "
             "later DMA issue times are unspecified.",fontsize=8.8,color=sub)
    fig.text(.20,.044,"CA uses GM/UB descriptor windows and the first-to-last positive-duration conversion on each tile. "
             "Slot IDs follow the observed UB addresses.",fontsize=8.8,color=sub)
    fig.savefig(path,dpi=155);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-root",type=Path,help="Checkout retaining runs/vector21 and its qualified receipts")
    args=parser.parse_args()
    for tag,chip in (("dt","Ascend950DT_9582"),("pr","Ascend950PR_957b")):
        out=OUT/("slot-figure-"+tag+".json")
        if args.raw_root:
            raw=args.raw_root.resolve()
            freeze_path=ROOT/"results/cast-capacity/two-slot-freeze.json"
            freeze=read(freeze_path)
            job,=[j for j in freeze["included"] if j["shape"]==SHAPE]
            row,=[r for r in read(raw/"results/v0.21/vector/holdout.json") if
                  r["id"]==job["id"] and r["chip"]==chip and r["qualified"]]
            assert row["cost"]["started_utc"]>freeze["frozen_utc"]
            model,ca=predicted(row,job["targets"][chip]),measured(raw,row)
            assert (ca["active_aic"],ca["active_aiv"]) == (0,model["plan"]["active_aiv"])
            assert ca["correctness"] == [f"correctness count={math.prod(SHAPE)} bad=0"]
            assert [t["n"] for t in model["tiles"]]==[t["n"] for t in ca["tiles"]]
            out.write_text(json.dumps(dict(sku_short="950"+tag.upper(),shape=SHAPE,dtype="bf16",
                                      freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                                      freeze_utc=freeze["frozen_utc"],ca_started_utc=row["cost"]["started_utc"],
                                      model=model,ca=ca),
                                      indent=2,allow_nan=False)+"\n")
        render(read(out),ROOT/f"docs/examples/figures/v022-cast-ub-{tag}.png")


if __name__=="__main__":
    main()
