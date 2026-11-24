"""Regression checks for disjoint source/consumer work and native CA evidence."""

import json
from pathlib import Path
from types import SimpleNamespace

import unittest

from operator_model.chips import load_chip
from operator_model.system_routes import attention

DT = load_chip("Ascend950DT_9582")
PR = load_chip("Ascend950PR_957b")
RECEIPTS = Path(__file__).resolve().parents[1] / "calibration/system_attention_ca"


def tensor(shape, dtype="bfloat16"):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype)


def node(raw, inputs, outputs, **sem):
    return SimpleNamespace(
        raw_name=raw,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        attributes=(("semantic", tuple(sem.items())),),
    )


def conv(width):
    return node(
        "vllm_ascend::causal_conv1d_update_reference",
        [tensor([9, width]), tensor([4, width]), tensor([1, 11, width])],
        [tensor([9, width])],
        activation="silu",
    )


def fia(length):
    return node(
        "npu::npu_fused_infer_attention_score",
        [
            tensor([9, 12, 64]),
            tensor([9, 2, 64]),
            tensor([9, 2, 64]),
            tensor([1, length - 9, 2, 2, 64]),
        ],
        [tensor([9, 12, 64]), tensor([12, 9], "float32")],
        kv_lens=(length,),
        effective_qk_pairs=9 * length,
        query_lens=(9,),
        num_kv_heads=2,
        compute_precision="bf16",
    )


def indexer(layer):
    return node(
        "inference::lightning_indexer",
        [
            tensor([9, 32, 128]),
            tensor([1, 65540, 128]),
            tensor([9, 32]),
            tensor([9], "int64"),
        ],
        [tensor([9, 512], "int32")],
        index_topk=512,
        kv_source=layer,
        index_source=layer,
        reference="vllm_ascend/models/deepseek_v41/model.py:DeepseekV41Attention",
    )


class SystemAttentionTests(unittest.TestCase):
    def test_frozen_native_predictions(self):
        for row in json.loads((RECEIPTS / "frozen.json").read_text())["cases"]:
            features = {"fia": fia, "conv": conv}[row["operator"]](row["extent"])
            result = attention.price(features, load_chip(row["sku"]))
            self.assertAlmostEqual(result.latency_us, row["model_us"], places=10)

    def test_native_receipts_separate_sku_and_full_window(self):
        holdout = json.loads((RECEIPTS / "independent.json").read_text())
        frozen = json.loads((RECEIPTS / "frozen.json").read_text())["cases"]
        self.assertEqual(
            {(x["operator"], x["sku"]) for x in holdout},
            {(x["operator"], x["sku"]) for x in frozen},
        )
        for row in holdout:
            self.assertAlmostEqual(
                row["device_us"], sum(x["cycles"] for x in row["stages"]) / 1650
            )
            self.assertTrue(all(len(x["elf_sha256"]) == 64 for x in row["stages"]))
        for row in json.loads((RECEIPTS / "mxindexer-static.json").read_text()):
            schedule = row["stages"][0]["single_shard_schedule"]
            self.assertEqual(
                (schedule["mmad_mx_count"], schedule["issuing_aic"]), (17, 1)
            )
            self.assertEqual(schedule["mmad_mnk"], [64, 128, 128])
        self.assertTrue(
            all(
                x["device_us"] is None
                for x in json.loads(
                    (RECEIPTS / "metadata-eligibility.json").read_text()
                )
            )
        )

    def test_candidates_rank_all_scores_once(self):
        ordinary = attention.price(indexer(8), DT)
        producer = attention.price(indexer(20), DT)
        for item, expected in ((ordinary, False), (producer, True)):
            assert (
                item.work["schedule"]["stage_budget"]["layer20_produces_block_ids"]
                == expected
            )
        assert ordinary.latency_us < producer.latency_us
        publish = node(
            "inference::candidate_blocks",
            [tensor([9, 512], "int32")],
            [tensor([9, 2048], "int32")],
            block_size=8,
            source_layer=20,
        )
        work = attention.price(publish, DT).work
        assert work["cube_ops"] == 0
        assert work["logical_write_bytes"] == 9 * 2048 * 4
        assert work["schedule"]["tile"]["width"] == 2048

    def test_gqa_counts_valid_keys_once_across_query_heads(self):
        work = attention.price(fia(512), DT).work
        assert work["cube_ops"] == 2 * 9 * 512 * 12 * (64 + 64)
        assert work["logical_read_bytes"] == 2 * (9 * 12 * 64 + 512 * 2 * (64 + 64))
        assert work["physical_hbm_bytes"] is None

    def test_delta_state_stays_resident_across_verification(self):
        inputs = [tensor([1, 9, 12, 64], "float4_e2m1fn_x2")] * 3 + [
            tensor([1, 9, 12, 2, 2], "float8_e8m0fnu")
        ] * 3
        inputs += [
            tensor([1, 9, 12, 128]),
            tensor([1, 9, 12], "float32"),
            tensor([12], "float32"),
            tensor([1536], "float32"),
            tensor([1, 12, 128, 128], "float32"),
        ]
        x = node(
            "vllm_ascend::recurrent_kda_reference",
            inputs,
            [tensor([1, 9, 12, 128])],
            compute_precision="fp4",
            state_layout="BHVK",
        )
        work = attention.price(x, DT).work
        assert work["schedule"]["tile"]["serial_steps"] == 9
        assert work["logical_write_bytes"] == 4 * 12 * 128 * 128 + 2 * 9 * 12 * 128

    def test_unknown_precision_is_rejected(self):
        f = fia(512)
        f.attributes = (
            (
                "semantic",
                tuple(
                    dict(
                        dict(f.attributes)["semantic"], compute_precision="fp4"
                    ).items()
                ),
            ),
        )
        with self.assertRaises(ValueError):
            attention.price(f, PR)
