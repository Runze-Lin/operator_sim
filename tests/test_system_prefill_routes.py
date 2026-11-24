"""Kimi long-context prefill uses chunk-local tensors and persistent sequence state."""

from types import SimpleNamespace
import unittest

from operator_model.chips import load_chip
from operator_model.system_routes import estimate_full_features
from operator_model.system_routes.prefill import (
    BF16,
    CONV,
    COPY,
    E8,
    FP32,
    FP4,
    INSERT,
    KDA,
    NORM,
    ZERO,
)
from operator_model.system_routes.prefill import price


def tensor(dtype, *shape):
    pitch = 1
    strides = []
    for size in reversed(shape):
        strides.append(pitch)
        pitch *= size
    return SimpleNamespace(
        dtype="torch." + dtype,
        shape=shape,
        stride=tuple(reversed(strides)),
        device="meta",
        layout="torch.strided",
    )


def node(name, ins, outs, **sem):
    return SimpleNamespace(
        raw_name=name,
        attributes=(("semantic", tuple(sem.items())),),
        inputs=ins,
        outputs=outs,
    )


def chunk(lengths, heads=12, depth=128):
    total = sum(lengths)
    f4 = tensor(FP4, 1, total, heads, depth // 2)
    scales = tensor(E8, 1, total, heads, depth // 64, 2)
    state = tensor(FP32, len(lengths), heads, depth, depth)
    output = tensor(BF16, 1, total, heads, depth)
    return node(
        KDA,
        (f4,) * 3
        + (scales,) * 3
        + (
            output,
            tensor(FP32, 1, total, heads),
            tensor(FP32, heads),
            tensor(FP32, heads * depth),
            state,
        ),
        (output, state),
        query_lens=lengths,
        group_size=32,
        qk_normalization_location="before_chunk",
        state_layout="BHVK",
        chunk_size=64,
        algorithm="chunk",
        compute_precision="fp4",
        accumulator_dtype=FP32,
    )


def cache(capacity, rows=2048):
    bank = tensor("uint8", 1, capacity, 68)
    return node(
        INSERT,
        (
            tensor(BF16, rows, 128),
            tensor(FP32, 128),
            tensor(FP32, 128),
            tensor("int64", rows),
            bank,
        ),
        (bank,),
        key_norm="layernorm",
        storage_precision="fp4",
        writes="current query positions only",
        logical_head_dim=128,
        row_bytes=68,
        query_lens=(rows,),
    )


class PrefillTest(unittest.TestCase):
    def setUp(self):
        self.chip = load_chip("Ascend950DT_9582")

    def test_real_prefill_geometry_with_full_adapter(self):
        y = tensor(BF16, 2048, 4608)
        state = tensor(FP32, 1, 12, 128, 128)
        x = tensor(BF16, 1, 2048, 12, 128)
        cases = [
            node(
                CONV,
                (y, tensor(BF16, 4, 4608), tensor(BF16, 1, 3, 4608)),
                (y,),
                kernel_size=4,
                phase="prefill",
                query_lens=(2048,),
                has_initial_state=(True,),
                activation="silu",
            ),
            node(NORM, (x,), (x,), normalization="l2", head_dim=128),
            chunk((2048,)),
            node(COPY, (state, state), (state,)),
            node(ZERO, (state,), (state,)),
            cache(33810),
        ]
        for case in cases:
            with self.subTest(op=case.raw_name):
                result = estimate_full_features(case, chip=self.chip)
                self.assertGreater(result.latency_us, 0)
                self.assertTrue(result.status == "extrapolated")
                self.assertIsNone(result.work["physical_hbm_bytes"])

    def test_cache_storage_is_written_at_new_positions_only(self):
        first = price(cache(33810), self.chip)
        longer = price(cache(526336), self.chip)
        self.assertEqual(first.latency_us, longer.latency_us)
        self.assertEqual(first.work["logical_write_bytes"], 2048 * 68)
        self.assertEqual(
            first.work["logical_read_bytes"], 2048 * (256 + 8) + 2 * 128 * 4
        )

    def test_zeros_and_copy_are_distinct_writes(self):
        state = tensor(FP32, 4, 12, 128, 128)
        n = 4 * 4 * 12 * 128 * 128
        for name, inp, read in ((COPY, (state, state), n), (ZERO, (state,), 0)):
            w = price(node(name, inp, (state,)), self.chip).work
            self.assertEqual(
                (w["logical_read_bytes"], w["logical_write_bytes"]), (read, n)
            )

    def test_chunk_precision_tail_and_serial_state(self):
        for lens, serial, squares in [
            ((63,), 1, 63**2),
            ((64,), 1, 64**2),
            ((65,), 2, 64**2 + 1),
            ((128, 17), 2, 2 * 64**2 + 17**2),
        ]:
            result = price(chunk(lens, heads=2), self.chip)
            self.assertEqual(
                result.work["cube_ops_by_dtype"]["mxfp4"], 2 * 2 * 128 * squares
            )
            self.assertGreater(result.work["cube_ops_by_dtype"]["bf16"], 0)
            self.assertEqual(result.work["schedule"]["max_serial_chunks"], serial)
            self.assertEqual(result.work["schedule"]["state_accumulator"], "fp32")

    def test_unsupported_layout_does_not_receive_fp4_price(self):
        f = chunk((2048,))
        f.inputs = (tensor(BF16, 1, 2048, 12, 128), *f.inputs[1:])
        result = estimate_full_features(f, chip=self.chip)
        self.assertIsNone(result.latency_us)
        self.assertIn("expected float4", result.reason)
