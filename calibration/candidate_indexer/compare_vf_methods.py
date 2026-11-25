"""Same archived physical windows, both VF cost models. No collection, no fitting."""

import csv
import json
from pathlib import Path
from operator_model import estimate

HERE = Path(__file__).resolve().parent


def records():
    def read(key):
        return json.loads((HERE / (key + ".json")).read_text())

    def packed(shape, chip, ca, source, attrs):
        return dict(
            scope="native fused complete CA replay",
            chip=chip,
            shape=shape,
            dtype="mxfp4",
            attrs=dict(implementation="a5_qsli_projection", **attrs),
            ca_us=ca,
            source=source,
        )

    for r in read("v112_merge_replay")["records"]:
        yield packed(
            [1, 1, 32, 128, r["L"], 2048, 512],
            r["sku"],
            r["ca_us"],
            "v112_merge_replay",
            dict(candidate_limit=r["candidate_count"], table_pages=r["table_width"]),
        )
    for src, overlap in [("prefill-wave", 1.0), ("v112_wave_pr", 1.0), ("v112_wave_dt", 0.0)]:
        r = read(src)
        yield packed(
            r["shape"], r["sku"], r["device_us"], src, dict(table_pages=8201, candidate_order_overlap=overlap)
        )
    for arm in ["shared", "moving", "rotated"]:
        src = "v113_control_" + arm
        r = read(src)
        yield packed(
            r["shape"],
            r["sku"],
            r["device_us"],
            src,
            dict(table_pages=8201, candidate_order_overlap=r["logical_all_row_pair_same_rank"]),
        )
    for src in ("compact", "dense", "compact-component-check"):
        for r in read(src):
            shape = [1, r["query_rows"], r["heads"], r["head_dim"], r["compressed_keys"], r["selected"], 512]
            yield dict(
                scope="hypothetical fusion; comparison chain has different kernels",
                shape=shape,
                source=src,
                chip=r["chip"],
                dtype="bf16",
                ca_us=None,
                component_sum_us=r["sum_kernel_windows_us"],
                attrs=dict(
                    implementation="li_pipeline_projection",
                    score_policy="full_scan" if src == "dense" else "candidate_first",
                    block_size=8,
                ),
            )


def main(dest):
    rows = []
    for r in records():

        def quote(mode):
            x = estimate(
                "candidate_filtered_indexer",
                r["shape"],
                {**r["attrs"], "vector_model": mode},
                chip=r["chip"],
                dtype=r["dtype"],
            )
            if x.latency_us is None:
                raise ValueError(x.reason)
            return x.latency_us

        old, new = quote("calibrated"), quote("instruction_classes")
        rows.append(
            dict(
                **r,
                legacy_us=old,
                classes_us=new,
                legacy_error_pct=100 * (old / r["ca_us"] - 1) if r["ca_us"] else None,
                classes_error_pct=100 * (new / r["ca_us"] - 1) if r["ca_us"] else None,
            )
        )
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=(
                "scope",
                "source",
                "chip",
                "dtype",
                "shape",
                "attrs",
                "ca_us",
                "component_sum_us",
                "legacy_us",
                "classes_us",
                "legacy_error_pct",
                "classes_error_pct",
            ),
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "shape": json.dumps(r["shape"], separators=(",", ":")),
                    "attrs": json.dumps(r["attrs"], sort_keys=True, separators=(",", ":")),
                }
            )


if __name__ == "__main__":
    main(HERE.parents[1] / "results/v2.3/candidate-methods.csv")
