"""Regression checks on the local MLA projections and routed MoE boundaries."""

from dataclasses import replace
import unittest

from operator_model import load_chip
from operator_model.system_routes import estimate_full_features
from test_system_adapter import BF16, FP8, F32, Features, attributes, tensor


class CatalogCubeTests(unittest.TestCase):
    def price(self, f):
        return estimate_full_features(f, chip=load_chip("Ascend950DT_9582"))

    def test_fp8_k_row_scale_bmm(self):
        g, m, k, n = 12, 4, 128, 512
        x, w = (g, m, k), (g, k, n)
        sem = dict(
            group_size=32,
            accumulator_dtype=F32,
            compute_precision="fp8",
            input_layout="values_then_corresponding_scales",
            logical_input_shapes=(x, w),
        )
        f = Features(
            "inference::quantized_batched_matmul",
            (
                tensor(FP8, x, (k, g * k, 1)),
                tensor(FP8, w),
                tensor(F32, (g, m)),
                tensor(F32, (g, k)),
            ),
            (tensor(BF16, (g, m, n)),),
            attributes(sem),
        )
        p = self.price(f)
        self.assertEqual(p.work["cube_ops_by_dtype"], {"bf16": 2 * g * m * k * n})
        self.assertEqual(p.work["logical_read_bytes"], g * (k * (m + n) + 4 * (m + k)))
        self.assertEqual(p.work["schedule"]["vector_model"], "instruction_classes")
        self.assertIsNone(
            self.price(
                replace(f, inputs=(*f.inputs[:3], tensor(F32, (g, n))))
            ).latency_us
        )
        self.assertIsNone(
            self.price(
                replace(f, inputs=(tensor(FP8, x, (1, 1, 1)), *f.inputs[1:]))
            ).latency_us
        )

    def test_fused_local_expert_counts(self):
        g, m, k, h, n = 4, 16, 256, 128, 512
        x, gate, down = (
            tensor(BF16, (m, k)),
            tensor(BF16, (g, 2 * h, k)),
            tensor(BF16, (g, n, h)),
        )

        def projection(shape):
            return tuple(
                dict(
                    logical_weight_shape=shape,
                    scheme="bf16",
                    weight_precision="bf16",
                    activation_precision="bf16",
                ).items()
            )

        roles = tuple(
            (("index", i), ("role", r))
            for i, r in enumerate(("activation", "gate_up_weight", "down_weight"))
        )

        def feature(counts):
            sem = dict(
                group_counts=counts,
                logical_input_shape=(m, k),
                workspace_bytes=None,
                fusion_components=("grouped_matmul", "swiglu", "grouped_matmul"),
                gate_up=projection(gate.shape),
                down=projection(down.shape),
                intermediate_quantization="none",
                internal_hbm_intermediates="not_materialized; workspace is separately declared",
                activation=(("kind", "silu"), ("limit", None)),
            )
            return Features(
                "inference::fused_grouped_mlp",
                (x, gate, down),
                (tensor(BF16, (m, n)),),
                attributes(sem, tensor_semantics=(("inputs", roles),)),
            )

        p = self.price(feature((8, 0, 0, 8)))
        self.assertEqual(p.work["cube_ops"], 2 * m * h * (2 * k + n))
        self.assertEqual(
            p.work["logical_read_bytes"], 2 * (m * k + 2 * (2 * h * k + n * h))
        )
        self.assertEqual(p.work["logical_write_bytes"], 2 * m * n)
        self.assertIsNone(self.price(feature((1, 2, 3, 4))).latency_us)

    def test_native_mx8_two_block_scales(self):
        from operator_model import estimate
        from operator_model.system_adapter import E8

        for m, n, k in (
            (1, 128, 64),
            (9, 256, 128),
            (11, 8192, 1536),
            (16, 512, 192),
            (32, 1280, 256),
            (63, 1536, 1024),
            (128, 512, 2048),
            (256, 768, 4096),
            (512, 2048, 7168),
            (4096, 2048, 2560),
        ):
            pages = (k + 63) // 64
            for chip in ("Ascend950DT_9582", "Ascend950PR_957b", "Ascend970DT"):
                results = [
                    estimate(
                        "matmul",
                        (m, n, k),
                        dict(weight_dtype="mxfp8", vector_model=mode),
                        chip=chip,
                        dtype="mxfp8",
                    )
                    for mode in ("calibrated", "instruction_classes")
                ]
                self.assertEqual(results[0].latency_us, results[1].latency_us)
                q = results[0]
                self.assertEqual(q.status, "extrapolated")
                self.assertEqual(
                    q.work["logical_read_bytes"], (m + n) * (k + 2 * pages)
                )
                self.assertEqual(
                    q.work["cube_ops_by_dtype"], {"fp8_e4m3fn": 2 * m * n * k}
                )
                self.assertEqual(q.work["vec_ops"], 0)
                self.assertGreater(q.latency_us, 0)
                self.assertGreaterEqual(q.projection["target_plan"]["k_pages"], 1)
                if k == 7168:
                    self.assertGreater(q.projection["target_plan"]["k_pages"], 1)
        m, n, k = 9, 256, 128
        roles = tuple(
            (("index", i), ("role", r))
            for i, r in enumerate(
                ("activation", "weight", "weight_scale", "activation_scale")
            )
        )
        sem = dict(
            activation_precision="fp8",
            weight_precision="fp8",
            activation_scale_layout="mx_e8m0",
            scale_layout="mx_e8m0",
            group_size=32,
            padding=(0, 0),
            accumulator_dtype=F32,
            logical_input_shape=(m, k),
            logical_weight_shape=(n, k),
        )
        f = Features(
            "npu::npu_quant_matmul",
            (
                tensor(FP8, (m, k)),
                tensor(FP8, (k, n), (1, k)),
                tensor(E8, (2, n, 2), (2, 4, 1)),
                tensor(E8, (m, 2, 2)),
            ),
            (tensor(BF16, (m, n)),),
            attributes(sem, tensor_semantics=(("inputs", roles),)),
        )
        self.assertEqual(self.price(f).contract, "matmul/A8W8_MX")
        self.assertIsNone(
            self.price(replace(f, inputs=(*f.inputs[:3], tensor(F32, (m,))))).latency_us
        )
