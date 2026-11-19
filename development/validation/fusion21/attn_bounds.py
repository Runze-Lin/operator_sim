"""Audit the allocated tensor ranges and failure ordering of the first registered 7168 test."""
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path(__file__).resolve().parents[3]
raw=ROOT/'runs/fusion21'
host=ROOT/'development/workloads/attention_residual_smoke.cpp'
kernel=ROOT/'runs/workloads19/sources/full/csrc/attention/attn_res_fwd/op_kernel/arch35/attn_res_fwd_resident.h'
assert '1e-5,N,S*H,N,1e-5,true,true,true,false' in host.read_text()
assert 'tiling_->blockTokenStride' in kernel.read_text() and 'tiling_->blockWriteIdx' in kernel.read_text()
t,h,slots,n=2,7168,8,3
buffers=dict(bank=dict(allocated=2*t*slots*h,largest_history_read_end=2*((t-1)*slots+n)*h,
                       largest_slot_write_end=2*((t-1)*slots+n+1)*h),
             prefix_addend_and_each_output=dict(allocated=2*t*h,largest_end=2*((t-1)*h+h)),
             each_shared_weight=dict(allocated=2*h,largest_end=2*h))
for buf in buffers.values():assert all(a<=buf['allocated'] for a in buf.values())
logs=[]
for chip in ('Ascend950DT_9582','Ascend950PR_957b'):
    p=raw/f'attnres_h7168_2_3_{chip}'/'run.log'
    lines=p.read_text().splitlines()
    def first(token):
        return next((i+1 for i,v in enumerate(lines) if token in v),None)
    logs.append(dict(chip=chip,path=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest(),
                     sigint_line=first('SigIntHandler received signal'),ldst_error_line=first('invalid ldst addr'),
                     timeout_line=first('The timeout has reached'),golden_line=first('correctness shape=')))
data=dict(shape=[t,h,slots,n],host_sha256=sha256(host.read_bytes()).hexdigest(),resident_kernel_sha256=sha256(kernel.read_bytes()).hexdigest(),
          buffer_bytes=buffers,token_stride_elements=slots*h,logs=logs,
          interpretation='Dense BF16 token-major bank stride is S*H elements in both caller and kernel. All analytically checked GM range ends are within allocations. A time cap and absence of golden are established, not an invalid kernel access before shutdown. Bounds do not prove instructions completed or simulator support.')
(ROOT/'results/v0.21/fusion/attn-bounds.json').write_text(json.dumps(data,indent=2)+'\n')
print(buffers,logs)
