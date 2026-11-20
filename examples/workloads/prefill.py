"""Export rank-local text-prefill contracts without allocating model tensors."""
import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from operator_model import estimate

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS = ("DeepSeek-V4.1-Flash", "DeepSeek-V4-Pro", "Kimi-K3")
CHIPS = ("Ascend950DT_9582", "Ascend950PR_957b", "Ascend970DT")
NOT_PRICED = "Serving fusion, layout, quantization or kernel schedule has no matching calibrated timing contract."


@lru_cache(maxsize=None)
def config(name):
    records = json.loads((MODELS_DIR / "sources.json").read_text())
    record = next(r for r in records if r["repo"].endswith("/" + name))
    raw = (MODELS_DIR / record["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Config digest changed for {name}")
    cfg = json.loads(raw)
    return cfg.get("text_config", cfg)


@dataclass(frozen=True)
class Scenario:
    model: str
    seq: int
    batch: int = 8
    tp: int = 8
    dp: int = 1
    ep: int = 8
    chunk: int = 0          # maximum tokens per attention replica, equally split among requests
    step: int = 0
    ep_rank: int = 0

    def __post_init__(self):
        c = config(self.model)
        if min(self.batch, self.seq, self.tp, self.dp, self.ep) < 1:
            raise ValueError("Positive batch, sequence and parallel dimensions required")
        if self.seq > c["max_position_embeddings"] or self.chunk < 0 or self.chunk % self.batch:
            raise ValueError("Unsupported sequence or per-replica chunk budget")
        e = c.get("n_routed_experts", c.get("num_experts"))
        if self.ep != self.tp * self.dp or e % self.ep or not 0 <= self.ep_rank < self.ep:
            raise ValueError("Scenario requires contiguous EP over all attention TP×DP ranks")
        if c["num_attention_heads"] % self.tp or c.get("o_groups", self.tp) % self.tp:
            raise ValueError("Attention heads and output groups must divide evenly across TP")
        if self.step < 0 or self.step >= self.steps:
            raise ValueError("Selected chunk step is outside this prompt")

    @property
    def stride(self):
        return self.chunk // self.batch if self.chunk else self.seq

    @property
    def steps(self):
        return (self.seq + self.stride - 1) // self.stride

    @property
    def prefix(self):
        return self.stride * self.step

    @property
    def q(self):
        return min(self.stride, self.seq - self.prefix)

    @property
    def t(self):
        return self.batch * self.q


@lru_cache(maxsize=None)
def price(op, shape, attrs):
    table = {}
    for chip in CHIPS:
        e = estimate(op, shape, json.loads(attrs), chip=chip, timing_mode="resource")
        table[chip] = {"us": e.latency_us, "status": e.status, "reason": e.reason}
    return table


def contracts(s, histogram=None):
    c = config(s.model)
    t, b, q, end = s.t, s.batch, s.q, s.prefix + s.q
    h, nl, heads = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    all_layers = list(range(nl))
    local_heads = heads // s.tp
    kimi = s.model == "Kimi-K3"
    flash = s.model == "DeepSeek-V4.1-Flash"
    rows = []

    def add(op, layers, tensors, *, evidence="logical component", reason=NOT_PRICED, call=None, **extra):
        layers = list(layers)
        if not layers:
            return
        row = {"op": op, "layers": layers, "tensors": tensors, "evidence": evidence, **extra}
        if "condition" in extra:
            row["mapping"] = "conditional runtime branch"
        elif op.startswith(("moe.routed_grouped_ffn","moe.dispatch_combine","moe.shared","parallel.","lm_head.")):
            row["mapping"] = "source geometry with explicit placement or scheduler assumptions"
        else:
            row["mapping"] = "shape derived from pinned source and config"
        if call:
            model_op, shape, *args = call
            attrs = args[0] if args else {}
            row["standalone_call"] = {"op": model_op, "shape": shape, "attrs": attrs}
            row["standalone_prediction"] = price(model_op, tuple(shape), json.dumps(attrs, sort_keys=True))
        else:
            row["unpriced_reason"] = reason
        rows.append(row)

    def linear(op, layers, k, n, m=t, groups=None, **extra):
        # Logical weighted GEMM; weights are output-major as in PyTorch Linear.
        a, w, y = [m, k], [n, k], [m, n]
        if groups is not None:
            a, w, y = [m, groups, k], [groups, *w], [m, groups, n]
        add(op, layers, {"x": a, "weight_logical": w, "y": y},
            useful_flops=2 * m * n * k * (groups or 1), **extra)

    add("embedding.lookup", ["input"], {"ids": [t], "out": [t, h]}, placement="logical vocab; embedding shard padding is runtime-specific")
    if s.step == s.steps - 1:
        linear("lm_head.last_position", ["output"], h, c["vocab_size"] // s.tp, m=b,
               placement="logical vocab shard; one sampled position per request, no prompt logprobs or padding")
    if not kimi:
        d, r, g, ro = c["head_dim"], c["q_lora_rank"], c["o_groups"], c["o_lora_rank"]
        streams, dim = c["hc_mult"], c["index_head_dim"]
        ih, top = c["index_n_heads"], c["index_topk"]
        gl = g // s.tp
        for stage in ("attn", "ffn"):
            add(f"mhc.{stage}.pre", all_layers,
                {"residual": [t, streams, h], "fn": [2*streams+streams**2, streams*h],
                 "post": [t, streams], "mix": [t, streams, streams], "out": [t, h]},
                iterations=c["hc_sinkhorn_iters"], parameter_dtype="fp32", variant="hc_pre_v3/pre_mix" if flash else "hc_pre_v2")
            add(f"mhc.{stage}.post", all_layers, {"x": [1,t,h], "residual": [1,t,streams,h],
                "post": [1,t,streams], "mix": [1,t,streams,streams], "out": [1,t,streams,h]},
                call=("mhc_post", [1,t,streams,h], {"implementation": "vllm_hc_post"}),
                evidence="vLLM _C_ascend.npu_hc_post unsqueezes the batch axis and calls aclnnHcPost")
        add("input_layernorm", all_layers, {"x": [t, h], "weight": [h], "out": [t, h]},
            call=("rmsnorm", [t, h]), evidence="standalone BF16 equivalent")
        add("post_attention_layernorm", all_layers, {"x": [t, h], "weight": [h], "out": [t, h]},
            call=("rmsnorm", [t, h]), evidence="standalone BF16 equivalent; placement may fuse")
        add("moe.router_input.float", all_layers, {"bf16": [t, h], "fp32": [t, h]},
            call=("cast_bf16_fp32", [t, h]), evidence="standalone BF16 to FP32 equivalent")
        linear("dsa.wq_a", all_layers, h, r)
        add("dsa.q_norm", all_layers, {"x": [t, r], "weight": [r], "y": [t, r]})
        linear("dsa.wq_b", all_layers, r, local_heads*d)
        add("dsa.q_norm_without_weight_and_rope", all_layers, {"q": [t, local_heads, d], "rope_suffix": [t, local_heads, c["qk_rope_head_dim"]]})
        linear("dsa.wkv", all_layers, h, d)
        add("dsa.kv_norm_rope_swa_cache", all_layers,
            {"current_kv": [t, 1, d], "weight": [d], "resident_window_logical": [b, min(end, c["sliding_window"]), 1, d]},
            **({"a5_swa_bytes_per_packed_row": 544} if flash else {}))
        linear("dsa.wo_a", all_layers, heads*d//g, ro, groups=gl)
        linear("dsa.wo_b", all_layers, gl*ro, h)
        roles = {}
        index_layers = []
        for i, ratio in enumerate(c["compress_ratios"][:nl]):
            if flash and ratio:
                source = max(j for j in c["kv_source_layer_ids"] if j <= i)
                index_source = max(j for j in c["index_source_layer_ids"] if j <= i)
            else:
                source = i if ratio else None
                index_source = i if ratio == 4 else None
            source_ratio = c["compress_ratios"][source] if source is not None else 0
            indexed = index_source is not None
            if indexed and index_source == i:
                index_layers.append(i)
            # All Pro layers own separate caches; Flash layers explicitly share caches and indices.
            key = (source_ratio, indexed, source if flash else "own", index_source if flash else "own" if indexed else None)
            roles.setdefault(key, []).append(i)
        for (ratio, indexed, source, ix), layers in roles.items():
            history = end // ratio if ratio else 0
            tensors = {"q": [t, local_heads, d], "out": [t, local_heads, d], "sink": [local_heads]}
            if ratio:
                tensors["compressed_kv_logical"] = [b, history, 1, d]
            if indexed:
                tensors["indices_capacity_per_query"] = [t, top]
            add("dsa.sparse_attention", layers, tensors, compression=ratio, kv_source=source, index_source=ix,
                max_window_keys=c["sliding_window"], max_compressed_keys_per_last_query=min(history, top) if indexed else history,
                reason="Sparse causal + SWA, optional compressed KV and sink; dense FA kernel would implement different attention.")
        for ratio in sorted({a for a in c["compress_ratios"][:nl] if a}):
            layers = ([i for i in c["kv_source_layer_ids"] if c["compress_ratios"][i] == ratio] if flash else
                      [i for i,rn in enumerate(c["compress_ratios"][:nl]) if rn == ratio])
            new = b*(end//ratio - s.prefix//ratio)
            coff = 2 if not flash and ratio == 4 else 1
            if flash and ratio == 2:
                add("compressor.input_cast", layers, {"bf16": [t,h], "fp32": [t,h]},
                    call=("cast_bf16_fp32",[t,h]), evidence="source hidden_states.float() before the two FP32 projections")
            for name in (["kv"] if flash and ratio == 1 else ["kv", "gate"]):
                linear(f"compressor.{name}", layers, h, coff*d)
            add("compressor.pool_store", layers,
                {"new_rows": [new,1,d], "history_logical": [b,end//ratio,1,d]},
                ratio=ratio, **({"a5_long_bytes_per_packed_row":320} if flash else {}))
            if not flash and ratio == 4:
                for name in ("kv", "gate"):
                    linear(f"indexer.compressor.{name}", layers, h, coff*dim)
                add("indexer.compressor.store", layers, {"new_rows": [new,1,dim], "history_logical": [b,end//ratio,1,dim]})
            if flash:
                linear("indexer.wk", layers, d, dim, m=t)
                add("indexer.norm_cache", layers, {"key_projection_rows": [t,1,dim], "new_unique_rows": [new,1,dim], "cache_logical": [b,end//ratio,1,dim]},
                    packed_k_bytes_per_row=64, packed_scale_bytes_per_row=4)
        linear("indexer.wq_b", index_layers, r, ih*dim, placement="replicated")
        linear("indexer.weights_proj", index_layers, h, ih, placement="replicated")
        if flash:
            add("indexer.project_weights.float", index_layers, {"bf16": [t,ih],"fp32": [t,ih]},
                call=("cast_bf16_fp32",[t,ih]), evidence="source weights_proj(hidden_states).float(); BF16 serving activations")
        add("indexer.weights_scale", index_layers, {"x": [t,ih],"scalar": [],"y": [t,ih]}, scalar=(dim*ih)**-0.5)
        if not flash:
            add("indexer.prepare_dsa_weights.float", index_layers, {"scaled_bf16": [t,ih], "fp32": [t,ih]},
                call=("cast_bf16_fp32",[t,ih]), evidence="A5DeviceAdaptor.prepare_dsa_indexer_weights after the BF16 projection and scalar multiply")
        add("indexer.select", index_layers, {"q": [t,ih,dim],"weights": [t,ih],"indices": [t,top]},
            reason="Packed sparse index kernels and candidate policy differ from standalone BF16 LightningIndexer.")
        if flash:
            candidate = c["candidate_source_layer_id"]
            add("indexer.candidate_blocks", [candidate], {"block_ids": [t,1,c["candidate_topk_blocks"]]}, block_size=c["candidate_block_size"])
            add("indexer.filtered_topk", [j for j in index_layers if j>candidate], {"block_ids": [t,1,c["candidate_topk_blocks"]],"indices": [t,top]})
            es = c["engram_layer_ids"]
            ew = (c["engram_max_ngram_size"]-1)*c["engram_n_heads"]*c["engram_head_dim"]
            add("engram.lookup", es, {"token_mask": [t], "rows": [t,ew]}, condition="engram_config enabled")
            linear("engram.wkv", es, ew, (streams+1)*h, condition="engram_config enabled")
            add("engram.gate", es, {"residual": [t,streams,h], "key": [t,streams,h], "value": [t,h], "out": [t,streams,h]}, condition="engram_config enabled")
        add("output.hc_collapse_and_norm" if flash else "output.hc_head_and_norm", ["output"],
            {"x": [t,streams,h],"out": [t,h]})
    else:
        kimi_contracts(c, s, all_layers, add, linear)

    if s.tp > 1:
        add("parallel.attention_row_reduce", all_layers, {"partial": [t,h], "result": [t,h]},
            reason="Row-parallel projection reduction; collective algorithm, fusion and topology are runtime inputs.")
    moe_contracts(c, s, all_layers, histogram, add, linear)
    return {"scenario": vars(s), "q_per_request": q, "prefix_per_request": s.prefix,
            "kv_after_per_request": end, "steps": s.steps, "tokens_per_rank": t,
            "maximum_position_per_request": c["max_position_embeddings"],
            "cu_seqlens": [i*q for i in range(b+1)], "activation_dtype": "bf16",
            "checkpoint_quantization": "compressed-tensors" if kimi else "fp8 (routed experts fp4)",
            "contracts": rows}


def kimi_contracts(c, s, all_layers, add, linear):
    t, h, b = s.t, c["hidden_size"], s.batch
    lc = c["linear_attn_config"]
    kda = [i-1 for i in lc["kda_layers"]]
    mla = [i-1 for i in lc["full_attn_layers"]]
    hd, nh = lc["head_dim"], lc["num_heads"]//s.tp
    x = nh*hd
    add("kda.input_projection", kda, {"x": [t,h], "qkv": [t,3*x], "beta": [t,nh],
        "raw_gate": [t,x], "output_gate": [t,x]},
        reason="Projection packing and mixed qkv/BFG overlap depend on checkpoint quant_method.")
    add("kda.mixed_input_dynamic_mx_quant", kda,
        {"bf16": [t,h], "fp8_e4m3fn": [t,h], "e8m0": [t,(h+63)//64,2]},
        condition="mixed QKV, MXFP8 dynamic method and scale_alg=0",
        evidence="conditional standalone BF16→E4M3FN, block 32",
        call=("quant",[t,h]))
    add("kda.causal_conv1d", kda, {"qkv": [t,3*x], "weight_packed": [lc["short_conv_kernel_size"],3*x],
        "out": [t,3*x]}, convolution_width=lc["short_conv_kernel_size"])
    add("kda.beta.sigmoid", kda, {"fp32_beta": [1,t,nh], "output": [1,t,nh]},
        call=("sigmoid_fp32",[t,nh]), evidence="standalone contiguous Beta sigmoid; mixed path computes once on auxiliary stream")
    shape = [1,t,nh,hd]
    add("kda.prefill.run_chunk_kda", kda, {"q": shape, "k": shape, "v": shape, "raw_gate": shape,
        "beta": [1,t,nh], "cu_seqlens": [b+1], "initial_state_vk": [b,nh,hd,hd],
        "final_state_vk": [b,nh,hd,hd], "output": shape}, chunk_size=64,
        reason="Variable-length packed requests, safe gate, state_v_first and output_final_state. A recurrent decode or dense equal-length chunk timing cannot price this call.")
    add("kda.gated_norm", kda, {"x": [t,nh,hd], "gate": [t,nh,hd], "out": [t,x]})
    linear("kda.o_proj", kda, x, h)
    nh = c["num_attention_heads"]//s.tp
    qr, kr = c["q_lora_rank"], c["kv_lora_rank"]
    nd, rd, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    linear("mla.fused_qkv_a_proj", mla, h, qr+kr+rd)
    add("mla.low_rank_norms", mla, {"q_rank": [t,qr], "kv_latent": [t,kr]}, reason="Often fused with MLA preparation.")
    linear("mla.q_b_proj", mla, qr, nh*(nd+rd))
    linear("mla.kv_b_proj", mla, kr, nh*(nd+vd))
    add("mla.causal_prefill", mla, {"q": [t,nh,nd+rd], "cache_logical": [b,s.prefix+s.q,1,kr+rd],
        "out": [t,nh,vd]}, qk_nope_width=nd, qk_rope_config_width=rd, applies_rotary=False,
        causal_pairs_per_local_head=b*(s.q*s.prefix+s.q*(s.q+1)//2),
        reason="Absorbed/paged or reconstructed KV and backend fusion are dispatch-dependent. No full square score tensor is allocated by this contract.")
    linear("mla.g_proj", mla,h,nh*vd)
    add("mla.output_gate", mla, {"out": [t,nh,vd],"gate": [t,nh,vd]})
    linear("mla.o_proj", mla,nh*vd,h)
    block = c["attn_res_block_size"]
    slots = (len(all_layers)+block-1)//block
    groups = {}
    for i in all_layers:
        older = (i+block-1)//block
        w = i % block == 0
        for tag,n,write,ad in (("attn",older,i//block if w else -1,i>0),
                               ("ffn",older+int(w),-1,not w)):
            groups.setdefault((tag,n,write,ad,True), []).append(i)
    groups[("output",slots,-1,True,False)] = ["output"]
    for (tag,n,write,ad,norm),layers in groups.items():
        attrs = {"optimize_prefill":True,"materialize":False,"write_idx":write,"addend":ad,"output_norm":norm}
        add(f"attn_res_fwd.{tag}", layers, {"prefix": [t,h], "bank": [t,slots,h], "output": [t,h]},
            valid_blocks=n, call=("attn_res_fwd",[t,h,slots,n],attrs), evidence="source fused op; work-only when branch outside calibration")
    f = c["intermediate_size"]//s.tp
    dense = all_layers[:c["first_k_dense_replace"]]
    linear("dense.gate_up", dense,h,2*f)
    add("dense.situ_and_mul", dense, {"x": [t,2*f], "out": [t,f]})
    linear("dense.down", dense,f,h)


def moe_contracts(c, s, all_layers, histogram, add, linear):
    t,h = s.t,c["hidden_size"]
    kimi = s.model == "Kimi-K3"
    layers = all_layers[c.get("first_k_dense_replace",0):]
    e = c.get("n_routed_experts", c.get("num_experts"))
    k = c.get("num_experts_per_tok", c.get("num_experts_per_token"))
    f = c["moe_intermediate_size"]
    el = e // s.ep
    hr = c.get("routed_expert_hidden_size",h)
    assignments = t*s.dp*k
    if histogram is None:
        local = [(i+1)*assignments//e - i*assignments//e for i in range(s.ep_rank*el,(s.ep_rank+1)*el)]
        hist_source = "synthetic balanced across global logical experts"
    else:
        if (len(histogram)!=e or any(type(i) is not int or not 0 <= i <= t*s.dp for i in histogram)
                or sum(histogram)!=assignments):
            raise ValueError("Histogram must contain one distinct-top-k assignment count per expert across all DP replicas")
        local = histogram[s.ep_rank*el:(s.ep_rank+1)*el]
        hist_source = "supplied global assignments; reused on listed layers"
    m = sum(local)
    linear("moe.gate_logits", layers,h,e, output_dtype="fp32")
    hash_layers = c.get("num_hash_layers",0)
    router = {"logits": [t,e], "weights": [t,k], "ids": [t,k]}
    add("moe.hash_router", [i for i in layers if i<hash_layers], {**router,"input_ids": [t]})
    add("moe.sigmoid_topk" if kimi else "moe.sqrtsoftplus_topk", [i for i in layers if i>=hash_layers],
        router, call=("moe_gating_top_k",[t,e,k]) if kimi else None,
        evidence="source semantic contract; separate standalone projection when supported")
    if hr != h:
        linear("moe.routed_down_transform", layers,h,hr)
        add("moe.routed_output_norm", layers, {"x": [t,hr], "y": [t,hr]}, reason="Conditional fused norm→MXFP8 when checkpoint selects this quantization.")
        linear("moe.routed_up_transform", layers,hr,h)
    add("moe.dispatch_combine", layers, {"rank_source_tokens": [t,hr],"logical_received_rows": [m,hr],
        "combined": [t,hr]}, communication="Routing, duplicate TP removal, padding and network schedule depend on mode.")
    add("moe.routed_grouped_ffn", layers, {"x": [m,hr],"w_gate_up_logical": [el,2*f,hr],
        "gate_up": [m,2*f],"activation": [m,f],"w_down_logical": [el,hr,f],"out": [m,hr]},
        local_expert_rows=local, histogram_source=hist_source, useful_flops=6*m*hr*f,
        activation=c["hidden_act"], reason="Physical expert pack, activation, router padding and selected grouped kernel require run metadata.")
    # Scenario chooses TP-sharded shared FFN, outside routed expert EP.
    sl = f*c.get("n_shared_experts",c.get("num_shared_experts",1))//s.tp
    linear("moe.shared_gate_up", layers,h,2*sl)
    add("moe.shared_activation", layers, {"gate_up": [t,2*sl],"out": [t,sl]}, activation=c["hidden_act"])
    linear("moe.shared_down", layers,sl,h)


def presets():
    for name in MODELS:
        for tp,dp in ((8,1),(1,8)):
            for n in (32768,131072,262144):
                for budget in (0,2048):
                    base = Scenario(name,n,tp=tp,dp=dp,chunk=budget)
                    yield base
                    if base.steps>1:
                        yield Scenario(name,n,tp=tp,dp=dp,chunk=budget,step=base.steps-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--seq", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0, help="total query tokens per replica dispatch; 0 means whole prompt")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--expert-counts", type=Path, help="JSON array of global assignment counts across DP replicas for a single scenario")
    args = parser.parse_args()
    s = Scenario(args.model,args.seq,args.batch,args.tp,args.dp,args.ep,args.chunk,args.step,args.ep_rank)
    hist = json.loads(args.expert_counts.read_text()) if args.expert_counts else None
    print(json.dumps(contracts(s,hist),indent=2))


if __name__ == "__main__":
    main()
