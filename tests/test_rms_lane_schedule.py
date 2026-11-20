from dataclasses import replace
import json
from pathlib import Path
import unittest

from operator_model import estimate, load_chip, OperatorStream
from operator_model.operators.rmsnorm.dispatch import price_rms
from operator_model.operators.rmsnorm.lane_schedule import parameters


ROOT = Path(__file__).resolve().parents[1]


class RmsLaneScheduleTests(unittest.TestCase):

    def test_local_ports_and_shared_floor(self):
        chip = load_chip()
        shape = (64,2048)
        def forecast(c):
            return estimate('rmsnorm', shape, {'resident_aiv':64}, chip=c)
        base = forecast(chip)
        for port in ('mte2', 'mte3'):
            slow = replace(chip, details={**chip.details, port+'_bandwidth_gbs':.5})
            r = forecast(slow)
            diag = r.projection['timing_diagnostics']
            floor = diag['lane_schedule']['port_floor_cycles'][port.upper()]
            self.assertGreaterEqual(r.latency_us*1000*chip.frequency_ghz, floor-1e-8)
            self.assertGreater(r.latency_us,base.latency_us)
        r = forecast(replace(chip, hbm_bandwidth_gbs=.5))
        d = r.projection['timing_diagnostics']
        self.assertEqual(d['lane_schedule']['critical_path'],'shared_io')
        self.assertEqual(set(r.projection['stages_us']),{'payload_release','shared_io'})
        self.assertAlmostEqual(r.projection['stages_us']['shared_io']*chip.frequency_ghz*1000,
                               d['shared_io_floor_cycles'])

    def test_waves_and_large_private_sector_ranges(self):
        chip = load_chip()
        r = estimate('rmsnorm',(8*32768,7168),{'resident_aiv':32},chip=chip)
        self.assertGreater(r.projection['target_plan']['iterations'],1)
        self.assertIsNone(r.projection['timing_diagnostics']['lane_schedule'])
        huge_ub = replace(chip, details={**chip.details,'ub_bytes':64*2**20})
        r = estimate('rmsnorm',(4_000_017,32),{'resident_aiv':32},chip=huge_ub)
        diag = r.projection['timing_diagnostics']
        self.assertEqual(diag['target_plan']['iterations'],1)
        self.assertLessEqual(diag['lane_schedule']['represented_sectors'],3*r.active_aiv)
        self.assertEqual(sum(diag['lane_schedule']['lane_class_counts'].values()),r.active_aiv)
