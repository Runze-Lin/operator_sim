from dataclasses import replace
import unittest

from operator_model import OperatorStream, estimate, load_chip
from operator_model.operators.mhc.history import estimate_post_history


OP = 'mhc_post'
DT = 'Ascend950DT_9582'
PR = 'Ascend950PR_957b'
A, B = (127, 4, 448), (157, 4, 1088)


def diag(f):
    return f.projection['timing_diagnostics']


class OrderedPostTest(unittest.TestCase):
    def test_first_then_changed_shape_reselects_by_target_and_reprices_rates(self):
        for name, state in ((DT, 'reuse_after_token'), (PR, 'expand_after_token'),
                            ('Ascend970DT', 'expand_after_token')):
            with self.subTest(name=name):
                stream = OperatorStream(name, track_post_history=True)
                first = stream.advance(OP, A)
                self.assertEqual(diag(first)['ordered_scenario'], 'first')
                other = stream.advance(OP, B)
                self.assertEqual(diag(other)['ordered_scenario'], state)
                self.assertAlmostEqual(other.latency_us,
                    estimate_post_history(B, chip=name).for_predecessor(first.active_aiv, A).latency_us)
                self.assertAlmostEqual(other.latency_us, sum(other.projection['stages_us'].values()))
                old = estimate(OP, B, {'resident_aiv': first.active_aiv}, chip=name)
                self.assertEqual(other.work, old.work)
                self.assertEqual(other.active_aiv, old.active_aiv)
                self.assertAlmostEqual(other.rates['counted_vector_tops'] * other.latency_us,
                                       old.rates['counted_vector_tops'] * old.latency_us)
                self.assertEqual(other.status, 'extrapolated')
                self.assertEqual(diag(stream.advance(OP, B))['model'], 'post_token_immediate_repeat')

    def test_option_validation_and_repeats_remain_independent(self):
        with self.assertRaises(TypeError):
            OperatorStream(track_post_history=1)
        s = OperatorStream(experimental_post_repeats=False, track_post_history=True)
        self.assertEqual(diag(s.advance(OP, A))['model'], 'post_token_ordered')
        self.assertEqual(s.advance(OP, A), estimate(OP, A, {'resident_aiv': 64}))
        self.assertEqual(diag(s.advance(OP, B))['ordered_scenario'], 'reuse_after_token')
        s.reset()
        ch = [3, 4, 192]
        for i in (0, 3):
            self.assertEqual(s.advance(OP, ch), estimate(OP, ch, {'resident_aiv': i}))
        self.assertEqual(diag(s.advance(OP, B))['model'], 'post_target_critical_aiv')  # no stage prior for expansion after channel

    def test_known_reuse_after_channel_and_unknown_prefix(self):
        s = OperatorStream(track_post_history=True)
        s.advance(OP, [16, 4, 768])  # 64 active AIVs
        self.assertEqual(diag(s.advance(OP, A))['ordered_scenario'], 'reuse_after_channel')
        s.forget_history()
        self.assertEqual(s.advance(OP, [32,4,256]), estimate(OP, [32,4,256]))
        self.assertEqual(s.preview(OP, A), estimate(OP, A))  # lower bound 32 < 64
        s.advance(OP, [40,4,256])
        self.assertEqual(diag(s.preview(OP, [33,4,256]))['ordered_scenario'], 'reuse_after_token')

    def test_explicit_context_foreign_kernel_and_failed_or_previewed_dispatch(self):
        s = OperatorStream(track_post_history=True)
        first = s.advance(OP, A)
        before = s.preview(OP, B)
        s.preview('matmul', [16,64,256])
        self.assertIsNone(s.advance(OP, [3,4,256], dtype='fp16').latency_us)
        with self.assertRaises(ValueError):
            s.advance(OP, [3,4,256], aic_only=True)
        self.assertEqual(s.preview(OP, B), before)
        for h in (0, None, first.active_aiv):
            self.assertEqual(s.preview(OP, B, {'resident_aiv': h}), estimate(OP, B, {'resident_aiv': h}))
        s.advance(OP, [3,4,256], {'resident_aiv': None})
        self.assertEqual(s.preview(OP, B), estimate(OP, B))
        s.advance(OP, A)
        s.advance(OP, [1,16,4,640], {'implementation': 'vllm_hc_post'})
        self.assertEqual(s.preview(OP, B), estimate(OP, B))
        s.reset()
        self.assertEqual(diag(s.preview(OP, B))['ordered_scenario'], 'first')

    def test_profile_footprint_and_stage_prices(self):
        c = load_chip()
        def run(chip, shape=A):
            return OperatorStream(chip, track_post_history=True).advance(OP, shape)
        self.assertEqual(diag(run(c,[65,4,5120]))['model'], 'post_target_critical_aiv')
        for custom in (replace(c, vector_tops={**c.vector_tops, '32': None}),
                       replace(c, details={**c.details, 'ub_bytes': 1024})):
            self.assertEqual(run(custom), estimate(OP, A, {'resident_aiv': 0}, chip=custom))
        slower = replace(c, frequency_ghz=c.frequency_ghz / 2)
        self.assertGreater(run(slower).latency_us, run(c).latency_us)
        bw = replace(c, details={**c.details, 'mte2_bandwidth_gbs': 1})
        self.assertGreater(diag(run(bw))['stages_us']['first_read_to_ready'],
                           diag(run(c))['stages_us']['first_read_to_ready'])
        vec = replace(c,vector_tops={**c.vector_tops, '32': c.vector_tops['32']/8})
        self.assertGreater(diag(run(vec))['stages_us']['ready_to_first_write'],
                           diag(run(c))['stages_us']['ready_to_first_write'])
