"""Switch VF service without changing dtype/work, wire or metadata contracts."""

from dataclasses import replace
from importlib.util import find_spec
from unittest import TestCase, skipUnless

from calibration.system_vector_ca.report import features
from operator_model import load_chip
from operator_model.system_routes import vf, vector
from operator_model.scheduling.transfer_pipeline import TransferProfile


class VectorMethodsTests(TestCase):
    def test_fused_swiglu_stages_and_scope(self):
        f = features("swiglu", (64, 7168))
        chip = load_chip()

        def quote(mode):
            with vf.selection(mode) as log:
                result = vector.price(f, chip)
            return result, log

        old, _ = quote("calibrated")
        new, stages = quote("instruction_classes")
        auto, _ = quote("auto")
        self.assertEqual(old.latency_us, auto.latency_us)
        self.assertEqual(old.work["logical_read_bytes"], new.work["logical_read_bytes"])
        self.assertIn("exp", stages[0]["counts"])
        self.assertEqual(new.work["schedule"]["active_aiv"], old.work["schedule"]["active_aiv"])
        self.assertGreater(
            new.work["schedule"]["first_panel_vector_cycles"],
            old.work["schedule"]["first_panel_vector_cycles"],
        )

    @skipUnless(find_spec("operator_performance_models"), "optional system platform")
    def test_backend_cache_identity_for_mode_and_port_source(self):
        from operator_model.sim_backend import FullOperatorSimBackend

        port = TransferProfile("DMA", "read", "ND", "bf16", 4096, 8, 24, "fixture")

        def conf(mode, response):
            cfg = replace(
                load_chip(), transfer_profiles={"read4096": replace(port, response_cycles=response)}
            )
            return FullOperatorSimBackend(chips={"custom": cfg}, vector_model=mode).configuration

        self.assertNotEqual(conf("calibrated", 24), conf("instruction_classes", 24))
        self.assertNotEqual(conf("calibrated", 24), conf("calibrated", 30))
        with self.assertRaises(ValueError):
            FullOperatorSimBackend(vector_model="V")
