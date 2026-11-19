from dataclasses import replace
from unittest import TestCase

from operator_model import estimate, load_chip


class Scenario970Test(TestCase):
    def test_peaks_buffers_and_sources(self):
        chip = load_chip('Ascend970DT')
        self.assertEqual(chip.cube_tops['4'], 8*chip.cube_tops['16'])
        self.assertAlmostEqual(chip.vector_tops['16'], 120*1.8*128*2*2/1000)
        self.assertEqual(chip.details['l2_bytes'], 200_000_000)
        self.assertEqual(chip.details['ub_bytes'], 248*1024)
        self.assertIn('details.ub_bytes',chip.inherited_fields)

    def test_pair_binding_and_extrapolation(self):
        chip = load_chip('Ascend970DT')
        r = estimate('fa',[9,32768,64],chip=chip)
        self.assertEqual((r.active_aic,r.active_aiv,r.status),(1,1,'extrapolated'))
        pooled = replace(chip, details={**chip.details,'fa_aiv_per_aic':2})
        p = estimate('fa',[9,32768,64],chip=pooled)
        self.assertEqual(p.active_aiv,2)
        for e in (p,r):
            wave,=e.projection['timing_diagnostics']['wave_types']
            self.assertLessEqual(wave['steady_kv_read_fraction_of_hbm_peak'],1.000001)
            self.assertEqual(wave['first']['steady_bottleneck'],
                             max(wave['first']['demands_cycles'],key=wave['first']['demands_cycles'].get))
        for bad in (0, 3, 1.5):
            with self.assertRaises(ValueError):
                replace(chip,details={**chip.details,'fa_aiv_per_aic':bad})
