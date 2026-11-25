"""The affine precision and row stride change the installed implementation contract."""

from math import ceil
import unittest
from types import SimpleNamespace
from operator_model.system_routes import estimate_full_features


def node(rows, width, stride, affine):
    def t(shape, dtype, strides):
        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            stride=strides,
            device="meta",
            layout="torch.strided",
        )

    x = t((rows, width), "bfloat16", (stride, 1))
    g = t((width,), affine, (1,))
    y = t(x.shape, x.dtype, (width, 1))
    return SimpleNamespace(
        raw_name="inference::indexer_layer_norm",
        inputs=(x, g, g),
        outputs=(y,),
        attributes=(),
    )


class CatalogLayerNorm(unittest.TestCase):
    def test_strides_and_reduction_distribution(self):
        for sku in ("Ascend950DT_9582", "Ascend950PR_957b"):
            for rows in (1, 9, 128):
                x = estimate_full_features(node(rows, 128, 160, "bfloat16"), chip=sku)
                self.assertEqual(x.work["schedule"]["materialize"], rows > 1)
                self.assertEqual(x.work["schedule"]["statistics_bytes"], 8 * rows)
                self.assertEqual(
                    x.work["schedule"]["active_aiv"],
                    ceil(rows / ceil(rows / (64 if "DT" in sku else 56))),
                )
                dense = estimate_full_features(
                    node(rows, 128, 128, "bfloat16"), chip=sku
                )
                self.assertFalse(dense.work["schedule"]["materialize"])
                self.assertEqual(x.work["cube_ops"], 0)
                self.assertEqual(x.work["logical_read_bytes"], 2 * rows * 128 + 4 * 128)
                if rows > 1:
                    self.assertGreater(x.latency_us, dense.latency_us)

    def test_fp32_affine_keeps_its_own_projection(self):
        x = estimate_full_features(
            node(4, 128, 192, "float32"), chip="Ascend950DT_9582"
        )
        self.assertIsNotNone(x.latency_us)
        self.assertIn("generic projection", x.work["schedule"]["evidence"])

    def test_prior_general_inputs(self):
        missing = node(4, 128, 128, "bfloat16")
        del missing.inputs[0].stride
        generic = estimate_full_features(missing, chip="Ascend950DT_9582")
        self.assertIsNotNone(generic.latency_us)
        self.assertNotIn("materialize", generic.work["schedule"])
        for shape, stride in (((2, 3, 128), (384, 128, 1)), ((0, 128), (128, 1))):
            f = node(1, 128, 128, "bfloat16")
            f.inputs[0].shape = f.outputs[0].shape = shape
            f.inputs[0].stride = f.outputs[0].stride = stride
            r = estimate_full_features(f, chip="Ascend950DT_9582")
            self.assertIsNotNone(r.latency_us)
            self.assertNotIn("materialize", r.work["schedule"])
