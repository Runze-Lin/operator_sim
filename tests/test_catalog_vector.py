from math import prod
from types import SimpleNamespace
import unittest

from operator_model.system_routes import estimate_full_features


def tensor(shape, dtype):
    return SimpleNamespace(
        shape=shape,
        stride=tuple(prod(shape[i + 1 :]) for i in range(len(shape))),
        dtype=dtype,
        device="meta",
        storage_offset=0,
    )


def quote(name, inputs, outputs, **semantic):
    f = SimpleNamespace(
        raw_name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        attributes=(("semantic", tuple(semantic.items())),),
    )
    return estimate_full_features(f, chip="Ascend950DT_9582")


class CatalogueVector(unittest.TestCase):
    def test_large_mhc_prefill_instruction_panels(self):
        from operator_model import estimate

        for rows in (257, 8192, 16384):
            value = estimate(
                "mhc_pre_sinkhorn",
                (rows, 4, 4096),
                {
                    "vector_model": "instruction_classes",
                    "num_iters": 20,
                    "need_backward": False,
                },
            )
            self.assertGreater(value.latency_us, 0)
            self.assertGreater(value.projection["stages_us"]["stage1_cube"], 0)

    def test_cache_capacity_vs_written_rows(self):
        payload = tensor((9, 512), "bfloat16")
        pos = tensor((9,), "int64")
        for capacity in (32768, 131072):
            cache = tensor((1, capacity, 512), "bfloat16")
            r = quote("inference::mla_cache_write", (payload, pos, cache), (cache,))
            self.assertEqual(r.work["logical_write_bytes"], 9 * 512 * 2)
            self.assertEqual(r.work["logical_read_bytes"], 9 * (512 * 2 + 8))

    def test_snapshots_price_output_and_do_not_requantize_history(self):
        def snapshot(cap):
            value = tensor((1, cap, 128), "float8_e4m3fn")
            scale = tensor((1, cap), "float32")
            return quote(
                "inference::cache_store",
                (tensor((9, 128), "bfloat16"), value, scale),
                (
                    tensor((1, 9, cap, 128), value.dtype),
                    tensor((1, 9, cap), scale.dtype),
                ),
                logical_cache_shape=value.shape,
                storage_precision="fp8",
                verify_state_strategy="paired encoded value/scale prefix snapshots",
                snapshot_query_len=9,
                snapshot_query_axis=1,
            )

        a, b = snapshot(512), snapshot(1024)
        self.assertEqual(a.work["logical_write_bytes"], 9 * 512 * 132)
        self.assertEqual(a.work["vec_ops"], b.work["vec_ops"])
        self.assertEqual(
            2 * a.work["logical_write_bytes"], b.work["logical_write_bytes"]
        )

    def test_layout_identity_requires_equal_physical_tensor(self):
        x = tensor((5, 256), "bfloat16")
        self.assertEqual(
            quote("inference::qkv_parallel_layout", (x,), (x,)).latency_us, 0
        )
        self.assertIsNone(
            quote(
                "inference::qkv_parallel_layout", (x,), (tensor((5, 128), x.dtype),)
            ).latency_us
        )

    def test_norm_fused_residual_outputs(self):
        x = tensor((17, 256), "bfloat16")
        g = tensor((256,), "bfloat16")
        rs = tensor((17, 1), "float32")
        r = quote("npu::npu_add_rms_norm", (x, x, g), (x, rs, x))
        self.assertEqual(r.work["logical_read_bytes"], 2 * (2 * 17 * 256 + 256))
        self.assertEqual(r.work["logical_write_bytes"], 2 * 2 * 17 * 256 + 4 * 17)
        self.assertIsNone(quote("npu::npu_add_rms_norm", (x, x, g), (x,)).latency_us)
