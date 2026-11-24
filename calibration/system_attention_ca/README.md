# 950DT/PR attention component receipts

Run from the root checkout. The compiler needs the installed 9.2.1 ACLNN headers and libraries. Simulator processes need local Unix sockets. Each invocation uses an empty output directory outside Git. Limit the number of simultaneous captures when CA shares a host.

```bash
sdk=toolchains/cann/9.2.1/cann-9.2.1/aarch64-linux
c++ -std=c++17 -O2 calibration/system_attention_ca/probe.cpp \
    -I"$sdk/include" -L"$sdk/lib64" -Wl,--allow-shlib-undefined \
    -lopapi -lnnopbase -lascendcl -o /tmp/att-probe
python -m calibration.system_attention_ca.capture scripts/cann-run.sh /tmp/att-probe \
    /tmp/ca-fia-950dt Ascend950DT_9582 fia 1024 --timeout-minutes 16
opp=toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950
python -m calibration.system_attention_ca.summarize "$opp" /tmp/ca-fia-950dt
```

| probe | final argument | check |
|---|---|---|
| `conv` | channels, even | All BF16 SiLU results and the entire stationary state for B1 Q9 K4. |
| `fia` | number of keys | Every grouped-head BF16 output and FP32 logsumexp in an unmasked contiguous GQA. |
| `indexer` | number of keys, at least 1032 | The top1024 set for each of nine queries. An explicit positive/negative score gap separates the sets. |
| `mxindexer` | number of keys, at least 2049 | Full V2 with actual AI CPU metadata generation. An empty table fails before the score kernel. |
| `mxsingle` | number of keys, at least 2049 | Alternative V2 score component with one explicitly enabled AIC in the installed 1024-word metadata protocol. Checks all selected indices, B1 Q1 H32 D128 top2048. |

`development.json` records the four response-time calibration points. `frozen.json` stores holdout predictions created before collection. `independent.json` records those CA runs. `indexer-h64.json` includes the ordinary H64 scoring component, SHA-256 and per-core MMAD runs. `mxindexer-static.json` records the 17 MMAD_MX instruction issues in the checked static plan. `metadata-eligibility.json` records the AI CPU limitation for automatic partitioning on the two SKUs. Probe dimensions differ from deployment memory layouts as described in the [model examples](../../docs/examples/system-attention-routes.md). Open-source V2 metadata describes a more recent dynamic table, while this installed 9.2.1 package implements the fixed table in `ops_transformer/ascendc/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h`.

The scorer requires the numerical golden, exact SKU/instrumentation settings, completion of all launched tasks, a complete core window and an ELF matching the installed operator package. A simulator returning an exit code of zero after a timeout still fails qualification. For repeated process executions use separate output directories; this collector handles a single native attention operator per run. Raw traces stay outside Git.
