import unittest
from dataclasses import replace
from types import SimpleNamespace

from operator_model import load_chip
from operator_model.system_routes import estimate_full_features


def args(rows, width=1024, *, gemma=False, dtype="bfloat16"):
    def t(shape, dtype=dtype):
        step = 1
        strides = []
        for n in reversed(shape):
            strides.append(step)
            step *= n
        return SimpleNamespace(
            shape=shape, stride=tuple(reversed(strides)), dtype=dtype, device="meta"
        )

    x = t((rows, width))
    return SimpleNamespace(
        raw_name="npu::npu_add_rms_norm",
        inputs=(x, x, t((width,))),
        outputs=(x, t((rows, 1), "float32"), x),
        attributes=(("semantic", (("gemma_weight_offset", gemma),)),),
    )


class AddRmsWindows(unittest.TestCase):
    def test_partition_and_shared_writers(self):
        for sku, active, owners in (
            ("Ascend950DT_9582", 64, 32),
            ("Ascend950PR_957b", 32, 16),
        ):
            r = estimate_full_features(args(64), chip=load_chip(sku))
            p = r.work["schedule"]
            self.assertEqual((p["active_aiv"], p["rstd_owners"]), (active, owners))
            self.assertEqual(p["rstd_drain_cycles"], 110 * (owners - 1))
            self.assertEqual(r.work["logical_write_bytes"], 64 * (4 * 1024 + 4))

    def test_evidence_boundary(self):
        for f in (args(4, gemma=True), args(4, dtype="float16"), args(4, width=32768)):
            r = estimate_full_features(f, chip=load_chip("Ascend950DT_9582"))
            self.assertIsNotNone(r.latency_us)
            self.assertNotIn("rstd_drain_cycles", r.work["schedule"])

    def test_custom_and_null_parameters(self):
        chip = load_chip("Ascend950PR_957b")
        overrides = dict(
            add_rms_rstd_service_bytes=64,
            add_rms_rstd_owner_cycles=27,
            add_rms_fullrow_setup_cycles=3000,
        )

        def predict(params):
            return estimate_full_features(
                args(64), chip=replace(chip, details={**chip.details, **params})
            )

        defaults = predict({})
        self.assertEqual(
            predict(dict.fromkeys(overrides)).latency_us, defaults.latency_us
        )
        plan = predict(overrides).work["schedule"]
        self.assertEqual(
            (plan["rstd_owners"], plan["setup_cycles"], plan["rstd_drain_cycles"]),
            (8, 3000, 7 * 27),
        )
