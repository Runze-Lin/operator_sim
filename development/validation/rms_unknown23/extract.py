"""Recover DMA and VF landmarks, elapsed from the first recorded instruction, per logical block."""
from collections import defaultdict
from pathlib import Path
import json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_dispatch_audit.schedule import launch_class
from development.validation.sku19.campaign import write
OUT=ROOT/'results/rms-unknown23'

def extract():
    archive=json.loads((ROOT/'results/rms-dispatch-audit/core-windows.json').read_text())['rows'];out=[]
    for r in archive:
        if any(len(b['stats'])!=1 for b in r['blocks']):continue
        db=sqlite3.connect((ROOT/r['source']).resolve().as_uri()+'?mode=ro',uri=True);by=defaultdict(dict)
        for c,l,pc,t,u,n in db.execute("select ExecInstrCoreId, CoreTypeId,SourceInstrAddr,ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName from ExecutedInstructions where ExecInstrName='VF' or ExecInstrName='MOV_SRC_TO_DST_ALIGNv2'"):
            by[c,l-2][pc]=[t,u]
        base=db.execute('select min(ExecInstrTickStart) from ExecutedInstructions').fetchone()[0];db.close()
        lanes=[];shape=r['shape'];t,w=shape;a=len(r['blocks']);h=r['prior'];stride=(t+63)//64
        for b in r['blocks']:
            state,part,local=launch_class(b['block'],a,h);ev=by[b['core'],b['lane']];k=b['block'];rows=min(stride,t-stride*k)
            fixed={0x9c,0xa78,0xc48,0x1064,0x1208,0x1138,0x1328}; first_vf,=set(ev)-fixed;expected=fixed|{first_vf}
            if set(ev)!=expected:raise ValueError((r['source'],k,sorted(ev)))
            e={hex(pc):[i-base,j-base] for pc,(i,j) in ev.items()}; arrival=b['begin'];g,ge=e['0xa78'];x,xe=e['0xc48'];vf,ve=e[hex(first_vf)]; _,sv=e['0x1064'];yp,ye=e['0x1328'];rs,rd=e['0x1138']
            assert rs==b['stats'][0]['issue'] and rd==b['stats'][0]['done'] and yp==b['stats'][0]['y_issue']
            lanes.append(dict(k=k,physical=b['core'],lane=b['lane'],state=state,first_vf_pc=hex(first_vf),part=part,local=local,rows=rows,arrival=arrival,front=g-arrival,g_latency=ge-g,x_gap=x-g,x_latency=xe-x,vf_gate=vf-max(xe,ge),stat_work=rs-vf,y_work=yp-vf,y_response=ye-yp,store_contention=rd-rs,retire=b['end']-max(ye,rd),g_issue=g,x_issue=x,rstd_issue=rs,y_issue=yp,last=b['end']))
        out.append(dict(shape=shape,H=h,active=a,role=r['role'],source=r['source'],window=r['window'],lanes=lanes))
    write(OUT/'landmarks.json',dict(source='v0.12 complete DT compact, no ELF attribution, one UB tile per AIV',rows=out));print(len(out),sum(len(r['lanes']) for r in out))
if __name__=='__main__':extract()
