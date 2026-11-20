"""No-history end-to-end Sinkhorn qualification and parity with premeasured forecasts."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from operator_model import OperatorStream, estimate, load_chip

ROOT=Path(__file__).resolve().parents[1]
OP='mhc_sinkhorn'
DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
FUT='Ascend970DT'


class TestUnknownSinkhorn(unittest.TestCase):
    def test_original_ten_shapes_every_price_already_sealed(self):
        jobs=json.loads((ROOT/'results/v0.26/sink/register.json').read_text())['jobs']
        for sku in (DT,PR,FUT):
            within=[r for r in jobs if r['chip']==sku and r['shape'][0]<=256]
            self.assertEqual(len({(*j['shape'],j['iters']) for j in within}),10)
            for j in within:
                args={'num_iters':j['iters']}
                public=estimate(OP,j['shape'],args,chip=sku)
                self.assertAlmostEqual(public.latency_us,j['unknown_us'],places=11)
                diag=public.projection['timing_diagnostics']
                self.assertAlmostEqual(sum(diag['stages_us'].values()),public.latency_us,places=12)
                self.assertNotIn('resident_aiv',diag['target_plan'])
                self.assertEqual(set(diag['scenario_us']),{'first','resident','expand'})
                self.assertEqual(public.status,'extrapolated' if sku==FUT else 'estimated')

    def test_unknown_explicit_none_and_known_family(self):
        attrs={'num_iters':10}
        p=estimate(OP,[67,4],attrs)
        self.assertEqual(estimate(OP,[67,4],dict(attrs,resident_aiv=None)),p)
        stream=OperatorStream(track_sinkhorn_history=True)
        cold=stream.advance(OP,[67,4],attrs);hot=stream.advance(OP,[67,4],attrs)
        self.assertGreater(cold.latency_us,p.latency_us)
        self.assertGreater(p.latency_us,hot.latency_us)
        self.assertEqual(cold,estimate(OP,[67,4],dict(attrs,resident_aiv=0)))
        self.assertEqual(hot.projection['target_plan']['history_state'],'resident')
        single=estimate(OP,[1,4],attrs).projection['timing_diagnostics']
        self.assertEqual(set(single['scenario_us']),{'first','resident'})
        self.assertLessEqual(single['modeled_range_us'][0],single['relative_minimax_us'])
        self.assertLessEqual(single['relative_minimax_us'],single['modeled_range_us'][1])

    def test_register_and_ub_boundaries_and_target_resources(self):
        for sku in (DT,PR,FUT):
            chip=load_chip(sku);last=chip.vector_cores*8
            self.assertIsNotNone(estimate(OP,[last,4],chip=sku).latency_us)
            beyond=estimate(OP,[last+1,4],chip=sku)
            self.assertEqual(beyond.status,'unavailable')
            self.assertIn('Multiple register',beyond.reason)
        c=load_chip(PR)
        slower=replace(c,name='lower-vector-rate',vector_tops={**c.vector_tops,'32':c.vector_tops['32']/2})
        self.assertGreater(estimate(OP,[67,4],chip=slower).latency_us,estimate(OP,[67,4],chip=c).latency_us)
        slow_hbm=replace(c,name='lower-bandwidth',hbm_bandwidth_gbs=.2)
        self.assertGreater(estimate(OP,[67,4],chip=slow_hbm).latency_us,estimate(OP,[67,4],chip=c).latency_us)
        small=replace(c,name='small-UB',details={**c.details,'ub_bytes':128})
        unpriced=estimate(OP,[67,4],chip=small)
        self.assertIsNone(unpriced.latency_us)
        missing=replace(c,name='unknown-vector',vector_tops={**c.vector_tops,'32':None})
        self.assertIsNone(estimate(OP,[67,4],chip=missing).latency_us)
        # The chip schema currently admits only the installed layout.
        with self.assertRaisesRegex(ValueError, 'Unknown chip detail'):
            replace(c,name='new-vreg',details={**c.details,'vector_register_bytes':512})
        self.assertEqual(estimate(OP,[257,4],chip=c,timing_mode='calibrated').status,'unavailable')
        self.assertEqual(estimate(OP,[67,4],{'return_intermediates':True}).status,'unsupported')
        self.assertEqual(estimate(OP,[67,4],dtype='bf16').status,'unsupported')
