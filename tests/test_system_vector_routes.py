import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

from operator_model.chips import load_chip
from operator_model.system_routes.vector import price


def tensor(shape, dtype, stride=None):
    if stride is None:
        stride = tuple(prod(shape[i+1:]) for i in range(len(shape)))
    return SimpleNamespace(shape=shape,stride=stride,dtype=dtype,layout="torch.strided",device="meta")


def node(raw, xs, ys, **sem):
    return SimpleNamespace(raw_name=raw,inputs=xs,outputs=ys,attributes=(("semantic",tuple(sem.items())),))


def test_cache_scatter_prices_only_updated_rows_with_and_without_positions():
    x=tensor((9,512),"bfloat16")
    for limit in (128,131072):
        values=tensor((1,limit,256),"float4_e2m1fn_x2")
        scales=tensor((1,limit,8,2),"float8_e8m0fnu")
        for positions in ((),(tensor((9,),"int64"),)):
            f=node("inference::cache_store",(x,*positions,values,scales),(values,scales),
                   logical_cache_shape=(1,limit,512),storage_precision="fp4")
            v=price(f,load_chip())
            assert v.latency_us>0
            assert v.work["logical_write_bytes"]==9*(256+16)
            assert v.work["logical_read_bytes"]==9*512*2+72*len(positions)


def test_frozen_trace_evidence_can_be_replayed_and_work_increases():
    from calibration.system_vector_ca.report import compare,features
    examples=json.loads((Path(__file__).resolve().parents[1]/"calibration/system_vector_ca/windows.json").read_text())
    for e in examples:
        assert abs(e["predicted_us"]-compare(e)["predicted_us"])<1e-8
    ref=load_chip()
    a=price(features("cache_load_fp8",(1,128,512)),ref)
    b=price(features("cache_load_fp8",(1,1024,512)),ref)
    assert b.work["logical_read_bytes"]==8*a.work["logical_read_bytes"]
    assert b.latency_us>a.latency_us


def test_rope_and_hash_touch_selected_slots():
    table=tensor((9,1,1,64),"bfloat16")
    f=node("_C_ascend::inplace_partial_rotary_mul",
           (tensor((9,1,8,512),"bfloat16"),table,table),(),partial_slice=(448,512))
    result=price(f,load_chip())
    assert result.work["logical_write_bytes"]==9*8*64*2
    r=node("_C_ascend::moe_gating_top_k_hash",(tensor((2,384),"float32"),tensor((2,),"int32"),
           tensor((129280,6),"int32")),(tensor((2,6),"float32"),tensor((2,6),"int32"),tensor((2,384),"float32")),
           out_flag=False)
    result=price(r,load_chip())
    assert result.work["logical_read_bytes"]==2*384*4+2*4+2*6*4
    assert result.work["logical_write_bytes"]==2*6*8


def test_wide_mhc_tracks_actual_projection_and_target():
    t,n,d=144,4,7168
    f=node("_C_ascend::npu_hc_pre_v2",(tensor((t,n,d),"bfloat16"),tensor((24,n*d),"float32"),
           tensor((3,),"float32"),tensor((24,),"float32")),(tensor((t,d),"bfloat16"),tensor((t,n),"float32"),
           tensor((t,n,n),"float32")),sinkhorn_iterations=20)
    for chip in ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT"):
        result=price(f,load_chip(chip))
        assert result.work["cube_ops"]==2*t*(n*d)*24 and result.work["vec_ops"]>0
        assert result.work["schedule"]["critical_m_panels"]==2


def test_slice_then_quant_counts_materialized_intermediate_and_empty_cache():
    shape=(9,128)
    f=node("npu::npu_dynamic_mx_quant",(tensor(shape,"bfloat16",(6288,1)),),
           (tensor((9,64),"float4_e2m1fn_x2"),tensor((9,2,2),"float8_e8m0fnu")),logical_shape=shape)
    result=price(f,load_chip())
    assert result.contract=="slice_then_quant_mxfp4"
    assert result.work["logical_read_bytes"]==2*2*9*128
    x=tensor((0,512),"bfloat16")
    v=tensor((1,32770,256),"float4_e2m1fn_x2")
    s=tensor((1,32770,8,2),"float8_e8m0fnu")
    empty=node("inference::cache_store",(x,v,s),(v,s),storage_precision="fp4",logical_cache_shape=(1,32770,512))
    result=price(empty,load_chip())
    assert result.latency_us==0 and result.work["logical_write_bytes"]==0
