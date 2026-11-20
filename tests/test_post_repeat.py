from dataclasses import replace
import unittest

from operator_model import OperatorStream, estimate, load_chip


class PostRepeatTest(unittest.TestCase):
    def test_default_stream_prices_only_an_exact_token_repeat(self):
        for sku in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
            with self.subTest(sku=sku):
                default = OperatorStream(sku)
                opt_in = OperatorStream(sku, experimental_post_repeats=True)
                legacy = OperatorStream(sku, experimental_post_repeats=False)
                # The ordinary first/changed-shape schedule is independent of repeat pricing.
                first = [127, 4, 448]
                second = [157, 4, 1088]
                resident = 0
                for shape in (first, second):
                    result = default.advance('mhc_post', shape)
                    self.assertEqual(result, legacy.advance('mhc_post', shape))
                    self.assertEqual(result, opt_in.advance('mhc_post', shape))
                    resident = max(resident, result.active_aiv)
                repeated = default.advance('mhc_post', second)
                self.assertEqual(repeated, opt_in.advance('mhc_post', second))
                previous = legacy.advance('mhc_post', second)
                self.assertEqual(repeated.projection['timing_diagnostics']['model'],
                                 'post_token_immediate_repeat')
                self.assertNotEqual(repeated.latency_us, previous.latency_us)
                self.assertEqual(previous, estimate('mhc_post', second,
                                 {'resident_aiv': resident}, chip=sku))

    def test_only_adjacent_success_is_eligible(self):
        chip = load_chip('Ascend950PR_957b')
        stream = OperatorStream(chip, experimental_post_repeats=True)
        shape = [67, 4, 384]
        cold = stream.advance('mhc_post', shape)
        self.assertEqual(cold.projection['timing_diagnostics']['ordered_scenario'], 'first')
        warm = stream.preview('mhc_post', shape, {'has_h_res': True}, dtype='bfloat16')
        self.assertEqual(warm.projection['timing_diagnostics']['model'], 'post_token_immediate_repeat')
        self.assertAlmostEqual(warm.latency_us, sum(warm.projection['stages_us'].values()))
        self.assertEqual(warm.work, cold.work)
        self.assertAlmostEqual(warm.rates['counted_vector_tops'] * warm.latency_us,
                               cold.rates['counted_vector_tops'] * cold.latency_us)
        # No state change from a dry run or from rejecting an invalid pre-dispatch query.
        stream.preview('mhc_post', [68, 4, 512])
        self.assertIsNone(stream.advance('mhc_post', shape, dtype='fp16').latency_us)
        with self.assertRaises(ValueError):
            stream.advance('mhc_post', shape, aic_only=True)
        self.assertEqual(warm, stream.advance('mhc_post', shape))
        self.assertEqual(stream.preview('mhc_post', shape, {'resident_aiv': 0}),
                         estimate('mhc_post', shape, {'resident_aiv': 0}, chip=chip))
        self.assertEqual(stream.preview('mhc_post', shape, {'resident_aiv': None}),
                         estimate('mhc_post', shape, chip=chip))
        stream.advance('matmul', [16,64,256])
        self.assertEqual(stream.preview('mhc_post', shape), estimate('mhc_post', shape, chip=chip))

    def test_context_reset_different_shape_and_channel(self):
        shape = [57,4,192]
        stream = OperatorStream(experimental_post_repeats=True)
        base = OperatorStream(experimental_post_repeats=False)
        for s in (base,stream):
            s.advance('mhc_post', shape)
        self.assertEqual(stream.preview('mhc_post',[56,4,192]),base.preview('mhc_post',[56,4,192]))
        base.reset()
        for s in (base,stream):
            s.reset()
            s.advance('mhc_post', [3,4,5376])
        self.assertEqual(stream.preview('mhc_post', [3,4,5376]),base.preview('mhc_post', [3,4,5376]))
        stream.forget_history()
        self.assertEqual(stream.preview('mhc_post',shape),estimate('mhc_post',shape))
        stream.advance('mhc_post',shape)
        self.assertEqual(stream.preview('mhc_post',shape).projection['timing_diagnostics']['model'],
                         'post_token_immediate_repeat')

    def test_distinct_successful_implementation_breaks_pair(self):
        stream = OperatorStream(experimental_post_repeats=True)
        shape = [69, 4, 640]
        stream.advance('mhc_post', shape)
        vllm=stream.advance('mhc_post', [1,69,4,640], {'implementation': 'vllm_hc_post'})
        self.assertIsNotNone(vllm.latency_us)
        self.assertEqual(stream.preview('mhc_post',shape),estimate('mhc_post', shape))

    def test_ub_boundary_changes_diagnostic_using_available_bytes(self):
        from operator_model.operators.mhc.repeat import channel_tiles
        chip = load_chip()
        self.assertEqual(channel_tiles(4864, chip)['count'], 1)
        self.assertEqual(channel_tiles(4880, chip)['count'], 2)
        small = replace(chip, details={**chip.details, 'ub_bytes': 65536})
        tile = channel_tiles(4096, small)
        self.assertEqual(tile['count'], 4)
        self.assertLess(320 + 52 * tile['inner'], 65536)
        self.assertEqual((tile['count'] - 1) * tile['inner'] + tile['tail'], 4096)
        wide = [65,4,5120]
        stream = OperatorStream(chip, experimental_post_repeats=True)
        stream.advance('mhc_post', wide)
        d = stream.preview('mhc_post', wide).projection['timing_diagnostics']
        self.assertEqual(d['channel_tiles']['inner'], 2560)
        self.assertEqual(d['schedule_granularity'], 'whole_row_aggregate')

    def test_missing_vector_peak_keeps_existing_estimate(self):
        chip = load_chip()
        chip = replace(chip,vector_tops={**chip.vector_tops,'32':None})
        stream = OperatorStream(chip,experimental_post_repeats=True)
        shape=[70,4,640]
        first=stream.advance('mhc_post',shape)
        self.assertEqual(stream.preview('mhc_post',shape),
                         estimate('mhc_post',shape,{'resident_aiv':first.active_aiv},chip=chip))
        with self.assertRaises(TypeError):
            OperatorStream(experimental_post_repeats=1)
