from dataclasses import replace
import json
from pathlib import Path
import unittest

from operator_model import RmsHistoryBand, estimate, estimate_rms_history_band, load_chip


ROOT = Path(__file__).resolve().parents[1]


class RmsHistoryBandTests(unittest.TestCase):

    def test_custom_port_prices_and_multi_ub_fallback(self):
        base = load_chip('Ascend950PR_957b')
        details = {**base.details, 'mte2_bandwidth_gbs': 3., 'mte3_bandwidth_gbs': 4.}
        chip = replace(base, name='custom-slow-ports', details=details)
        for shape in ([3, 1280], [563, 4048], [262144, 4096]):
            band = estimate_rms_history_band(shape, chip=chip)
            for high in (0, 1, band.active_aiv, chip.vector_cores):
                value = estimate('rmsnorm', shape, {'resident_aiv':high}, chip=chip)
                self.assertAlmostEqual(band.by_resident_aiv_us[min(high,band.active_aiv)], value.latency_us, places=11)

    def test_invalid_shape_and_dtype(self):
        for shape, dtype in (([2, 31], None), ([1, 256], 'fp16'), (None, None), ([False,16], None)):
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                estimate_rms_history_band(shape, dtype=dtype)
