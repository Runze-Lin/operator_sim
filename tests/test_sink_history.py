import json
from dataclasses import replace
from pathlib import Path
import unittest

from operator_model import OperatorStream, estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
FUT = 'Ascend970DT'


class SinkHistoryTests(unittest.TestCase):
    def test_all_frozen_issue_epochs_and_target_work(self):
        jobs = json.loads((ROOT/'results/v0.26/sink/register.json').read_text())['jobs']
        self.assertEqual(len(jobs), 45)
        for job in jobs:
            attrs = {'num_iters': job['iters'], 'resident_aiv': job['prior_h']}
            actual = estimate('mhc_sinkhorn', job['shape'], attrs, chip=job['chip'])
            expected = job['known']
            self.assertAlmostEqual(actual.latency_us, expected['latency_us'], places=12)
            self.assertEqual(actual.active_aiv, expected['active_aiv'])
            self.assertEqual(actual.work['logical_write_bytes'], expected['bytes_written'])
            self.assertEqual(actual.work['logical_read_bytes'], expected['bytes_read'])
            diag = actual.projection['timing_diagnostics']
            self.assertEqual(diag['target_plan']['vf_calls_per_core'], expected['vf_calls_per_core'])
            self.assertEqual(diag['target_plan']['history_state'], expected['selected'])
            for key, value in expected['vf_setup_epochs'].items():
                self.assertAlmostEqual(diag['issue_epochs_us'][key], value, places=12)
            self.assertAlmostEqual(sum(diag['stages_us'].values()), actual.latency_us, places=12)
            status = 'estimated' if job['shape'][0] <= 256 and job['chip'] in (DT, PR) else 'extrapolated'
            self.assertEqual(actual.status, status)

    def test_history_is_explicit_only(self):
        opts = {'num_iters': 10}
        stream = OperatorStream()
        base = estimate('mhc_sinkhorn', [67,4], opts)
        for attrs in (opts, {**opts,'resident_aiv':None}):
            self.assertEqual(stream.advance('mhc_sinkhorn', [67,4], attrs), base)
        first = estimate('mhc_sinkhorn', [67,4], {**opts,'resident_aiv':0})
        warm = estimate('mhc_sinkhorn', [67,4], {**opts,'resident_aiv':46})
        self.assertGreater(first.latency_us, warm.latency_us)
        self.assertEqual(stream.advance('mhc_sinkhorn', [67,4], opts), base)

    def test_exact_qualification_filters_and_custom_resources(self):
        chip = load_chip(PR)
        attrs = {'num_iters':10, 'resident_aiv':0}
        def result(t, target=chip, mode='auto', options=attrs):
            return estimate('mhc_sinkhorn', [t,4], options, chip=target, timing_mode=mode)
        self.assertEqual(result(67, mode='calibrated').status, 'estimated')
        outside = result(257)
        self.assertEqual(outside.status, 'extrapolated')
        qualified = result(257, mode='calibrated')
        self.assertIsNone(qualified.latency_us)
        self.assertIsNone(qualified.projection['latency_us'])
        self.assertEqual(qualified.status, 'unavailable')
        self.assertEqual(result(10, options={**attrs,'num_iters':100}).status, 'extrapolated')
        custom = replace(chip, name='future-variant')
        self.assertEqual(result(130,custom).latency_us, result(130).latency_us)
        self.assertEqual(result(130,custom).status,'extrapolated')
        self.assertGreater(result(130,replace(custom, hbm_bandwidth_gbs=.1)).latency_us, result(130).latency_us)
        slow_vec = replace(custom, vector_tops={**custom.vector_tops,'32':.1})
        self.assertGreater(result(130,slow_vec).latency_us, result(130).latency_us)
        slow_write = replace(custom, details={**custom.details,'mte3_bandwidth_gbs':.01})
        self.assertGreater(result(130,slow_write).latency_us, result(130).latency_us)
        low = replace(chip, details={**chip.details,'ub_bytes':1024})
        self.assertEqual(result(130,low).status, 'unavailable')
        absent = replace(chip, vector_tops={**chip.vector_tops,'32':None})
        self.assertIsNone(result(1,absent).latency_us)

    def test_only_valid_same_family_contract(self):
        for attrs in ({'resident_aiv':False},{'resident_aiv':57},{'resident_aiv':-1},
                      {'resident_aiv':1,'return_intermediates':True},{'resident_aiv':1,'num_iters':False},
                      {'resident_aiv':2,'same_shape':True}):
            self.assertEqual(estimate('mhc_sinkhorn',[64,4],attrs,chip=PR).status,'unsupported')
        self.assertEqual(estimate('mhc_pre_sinkhorn',[64,4,4096],{'resident_aiv':46}).status,'unsupported')
        self.assertEqual(estimate('mhc_sinkhorn',[1,4],{'resident_aiv':0},dtype='bf16').status,'unsupported')


if __name__ == '__main__':
    unittest.main()
