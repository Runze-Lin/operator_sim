"""Integration checks when the optional sim-workspace prediction API is installed."""
from dataclasses import replace
from importlib.util import find_spec
import unittest


@unittest.skipUnless(find_spec("operator_performance_models"), "requires sim-workspace")
class SystemBackendTests(unittest.TestCase):
    def setUp(self):
        from operator_performance_models import (
            BackendRouter, ChipCapabilities, ComputeCapability, ConstantUtilization,
            Hardware, HbmCapability, OpFeatures, TensorFeatures, create_backend,
        )

        util = ConstantUtilization(0.6)
        compute = ComputeCapability(((16, 500.), (32, 125.)), util)
        self.hardware = Hardware("950DT", chip=ChipCapabilities(
            compute, ComputeCapability(((16, 31.), (32, 16.)), util),
            HbmCapability(144., 4., util), head_tail_time=19.,
        ))
        def tensor(rows, cols):
            return TensorFeatures((rows, cols), "torch.bfloat16", "meta",
                                  (cols, 1), layout="torch.strided")
        self.node = OpFeatures("aten.mm.default", "Mm", (tensor(9, 512), tensor(512, 128)),
                               (tensor(9, 128),), category="cube")
        self.chain = ("operator_sim", "roofline")
        self.router = BackendRouter(create_backend(name) for name in self.chain)

    def test_hit_uses_device_window_and_system_resources(self):
        from operator_model import estimate

        prediction = self.router.predict(self.node, self.hardware, self.chain)
        backend, = self.router.resolve(("operator_sim",))
        target = backend._target_chip(self.hardware.name, self.hardware.chip)
        self.assertEqual((target.cube_tops["16"], target.vector_tops["32"],
                          target.hbm_bandwidth_gbs, target.cube_cores),
                         (500., 16., 4000., 32))
        self.assertIsNone(target.cube_tops["8"])
        self.assertEqual(prediction.attempted_backends, ("operator_sim",))
        reference = estimate("matmul", (9, 128, 512), chip=target)
        self.assertEqual(prediction.estimate.duration_us, reference.latency_us)
        details = prediction.estimate.details
        self.assertIn("extrapolated", details.assumptions)
        output = prediction.estimate.to_dict()
        self.assertEqual(output["total_cube_flops"], 2 * 9 * 128 * 512)
        self.assertIsNone(output["memory_access_time"])
        self.assertIsNone(output["head_tail_time"])

    @unittest.skipUnless(find_spec("sim_platform"), "requires SIMPlatform validation")
    def test_platform_validation_and_cache_roundtrip(self):
        from sim_platform.backend.prediction import validate_estimate
        from sim_platform.backend.prediction_codec import decode_prediction, encode_prediction

        prediction = self.router.predict(self.node, self.hardware, self.chain)
        validate_estimate(prediction.estimate)
        key = b"resource-schedule-test"
        self.assertEqual(decode_prediction(encode_prediction(prediction, key), key), prediction)
        other_details = replace(prediction.estimate.details, schema_version="other")
        with self.assertRaisesRegex(ValueError, "cube_us"):
            validate_estimate(replace(prediction.estimate, details=other_details))

    def test_explicit_future_template_participates_in_cache_identity(self):
        from operator_model import load_chip
        from operator_model.sim_backend import OperatorSimBackend

        base = replace(load_chip("Ascend970DT"), name="980-study")
        future = replace(self.hardware, name="my-980")
        backend = OperatorSimBackend(chips={"my-980": base})
        slower = OperatorSimBackend(chips={"my-980": replace(base, frequency_ghz=1.1)})
        self.assertNotEqual(backend.configuration, slower.configuration)
        result = backend.predict(self.node, future)
        self.assertIn("extrapolated", result.details.assumptions)
        self.assertNotEqual(result.duration_us, slower.predict(self.node, future).duration_us)

    def test_unknown_chip_and_unmatched_precision_keep_original_time(self):
        unknown = replace(self.hardware, name="untested")
        fp16 = replace(self.node,
                       inputs=tuple(replace(t, dtype="torch.float16") for t in self.node.inputs),
                       outputs=tuple(replace(t, dtype="torch.float16") for t in self.node.outputs))
        for node, hardware in ((self.node, unknown), (fp16, self.hardware)):
            with self.subTest(chip=hardware.name, dtype=node.inputs[0].dtype):
                predicted = self.router.predict(node, hardware, self.chain)
                baseline = self.router.predict(node, hardware, ("roofline",))
                self.assertEqual(predicted.attempted_backends, self.chain)
                self.assertEqual(predicted.backend_name, "roofline")
                self.assertEqual(predicted.estimate, baseline.estimate)
