from dataclasses import replace
import json
from pathlib import Path
import unittest

from operator_model import estimate, load_chip
from operator_model.vector_memory import stream_memory


class CastPressure(unittest.TestCase):
    def test_same_three_predictions_as_frozen_pair(self):
        reg=json.loads((Path(__file__).resolve().parents[1]/'results/v0.28/cast/register.json').read_bytes())
        for chip,scenarios in reg['prices'].items():
            pred=estimate('cast_bf16_fp32',(5632,5120),chip=chip)
            self.assertAlmostEqual(pred.latency_us,scenarios['pressure']['us'])
            self.assertEqual(pred.projection['timing_diagnostics']['retention_policy'],'pressure')

    def test_increasing_issue_capacity_changes_dirty_fraction_within_bounds(self):
        chip=load_chip('Ascend950PR_957b')
        n=5632*5120
        m=stream_memory(chip,chip.vector_cores,2*n,4*n,pressure_retention=True)
        limit=stream_memory(chip,1,2*n,4*n,pressure_retention=True)
        self.assertGreater(m['assumed_retained_write_bytes'],limit['assumed_retained_write_bytes'])
        self.assertEqual(m['modeled_external_bytes'],2*n+m['modeled_external_write_bytes'])
        self.assertLessEqual(m['assumed_retained_write_bytes'],min(4*n,chip.details['l2_bytes']))
        slow=replace(chip,details={**chip.details,'mte3_bandwidth_gbs':1})
        self.assertEqual(stream_memory(slow,chip.vector_cores,2*n,4*n,pressure_retention=True)['bandwidth_to_mixed_issue_ratio'],1.)
        self.assertEqual(estimate('sigmoid_fp32',(64,16),chip=chip).projection['timing_diagnostics']['retention_policy'],'uniform')
