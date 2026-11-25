"""Stride-independent invariants for latent, sparse and pooled projections."""

import unittest
from types import SimpleNamespace

from operator_model.chips import load_chip
from operator_model.system_routes import attention
from operator_model.system_routes.catalog_attention import _qsa_visits


def sparse_node(query, keys, budget):
    heads, latent, rope = 16, 512, 64

    def tensor(shape, dtype="bfloat16"):
        return SimpleNamespace(shape=shape, dtype=dtype)

    inputs = (
        tensor((query, heads, latent)),
        tensor((query, heads, rope)),
        tensor((1, keys, latent + rope)),
        tensor((query, budget), "int32"),
    )
    pairs = sum(min(v, budget) for v in range(keys - query + 1, keys + 1))
    sem = dict(
        logical_input_shapes=tuple(t.shape for t in inputs[:3]),
        compute_precision="bf16",
        query_lens=(query,),
        kv_lens=(keys,),
        selected_pair_bound=pairs,
    )
    return SimpleNamespace(
        raw_name="inference::sparse_mla",
        inputs=inputs,
        outputs=(tensor((query, heads, latent)),),
        attributes=(("semantic", sem),),
    )


class CatalogAttention(unittest.TestCase):
    def test_qsa_budget_and_frozen_tail(self):
        for q, kv, stop, ratio, budget in (
            (9, 32777, 32769, 4, 2048),
            (65, 65, 65, 3, 7),
            (1, 32, 24, 4, 8),
        ):
            visits = [min(x, stop) for x in range(kv - q + 1, kv + 1)]
            expected = sum(min(v // ratio * ratio, budget) + v % ratio for v in visits)
            self.assertEqual(_qsa_visits(q, kv, stop, ratio, budget), expected)

    def test_sparse_work_charged_only_to_selected_keys(self):
        for query, keys in ((1, 131072), (9, 32777), (4096, 32768), (65, 65)):
            f = sparse_node(query, keys, 2048)
            for sku in ("Ascend950DT_9582", "Ascend950PR_957b"):
                r = attention.price(f, load_chip(sku))
                pairs = dict(f.attributes)["semantic"]["selected_pair_bound"]
                self.assertEqual(r.work["cube_ops"], 2 * pairs * 16 * (576 + 512))
                self.assertEqual(r.status, "extrapolated")
                self.assertGreater(r.latency_us, 0)
                self.assertLessEqual(
                    r.work["schedule"]["active_aic"], load_chip(sku).cube_cores
                )

    def test_physical_dtype_conflict(self):
        f = sparse_node(1, 2048, 256)
        dict(f.attributes)["semantic"]["compute_precision"] = "fp4"
        with self.assertRaisesRegex(ValueError, "dtype"):
            attention.price(f, load_chip("Ascend950DT_9582"))
