import json
from dataclasses import replace
from pathlib import Path
import unittest

from operator_model import estimate,load_chip
from operator_model.mhc_sink_resource import fused as legacy

ROOT=Path(__file__).resolve().parents[1]
DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
FUT='Ascend970DT'


class FusedSinkDispatchTests(unittest.TestCase):
    def test_preregistered_partitions_and_price_parity(self):
        jobs=json.loads((ROOT/'results/v0.27/fusion/register.json').read_text())['jobs']
        shapes={tuple(j['shape']):j['predictions'] for j in jobs}
        self.assertEqual(len(shapes),11)
        for shape,pred in shapes.items():
            for chip,frozen in pred.items():
                r=estimate('mhc_pre_sinkhorn',shape,chip=chip)
                self.assertAlmostEqual(r.latency_us,frozen['candidate_us'],places=12)
                self.assertEqual(r.projection['target_plan'],frozen['target_plan'])
                self.assertEqual(set(r.projection['stages_us']),set(frozen['stages_us']))
                for name, value in frozen['stages_us'].items():
                    self.assertAlmostEqual(r.projection['stages_us'][name],value,places=12)
                self.assertEqual(r.status,'extrapolated' if chip==FUT else 'estimated')

    def test_future_and_distinct_tiling_preserve_existing_prices(self):
        attrs={'num_iters':20,'need_backward':False}
        for chip,t in ((FUT,72),(DT,128),(PR,129),(DT,257)):
            target=load_chip(chip)
            r=estimate('mhc_pre_sinkhorn',(t,4,4096),attrs,chip=target)
            self.assertEqual(r.latency_us,legacy((t,4,4096),target,attrs)[0])
            self.assertEqual(r.projection['timing_diagnostics']['schedule'],'prior_overlap_template')
            self.assertEqual(r.status,'extrapolated')
            self.assertIsNone(estimate('mhc_pre_sinkhorn',(t,4,4096),attrs,chip=target,timing_mode='calibrated').latency_us)
        c=replace(load_chip(PR),name='unseen-CV2')
        r=estimate('mhc_pre_sinkhorn',(80,4,4096),attrs,chip=c)
        self.assertEqual(r.status,'extrapolated')
        self.assertEqual(r.latency_us,estimate('mhc_pre_sinkhorn',(80,4,4096),attrs,chip=PR).latency_us)

    def test_validated_mode_and_original_interface_checks(self):
        shape=(72,4,4096)
        self.assertEqual(estimate('mhc_pre_sinkhorn',shape,chip=PR,timing_mode='calibrated').status,'estimated')
        for bad in ({'resident_aiv':46},{'num_iters':19},{'need_backward':True}):
            self.assertEqual(estimate('mhc_pre_sinkhorn',shape,bad).status,'unsupported')
        self.assertEqual(estimate('mhc_pre_sinkhorn',(72,4,2048)).status,'unsupported')


if __name__=='__main__':unittest.main()
