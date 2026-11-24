"""Regressions for reporting repeated, stateful windows and missing Roofline evidence."""
import unittest

from operator_model import estimate
from calibration.__main__ import encoded

from calibration.report_pages import DT, PR, render_contract, summary_ranges, reprice


class ErrorPageTests(unittest.TestCase):
    def test_main_summary_spans_all_comparable_windows_on_each_chip(self):
        header = '| 算子与输入合同（逐窗） | 原 CA 窗逻辑时间系数（倍，DT / PR） | 冷读写逻辑 U 范围（DT / PR） | WAPE |\n|---|---|---|---|\n'
        body = ('| [A](errors/a.md) | OLD | COLD | 2% |\n'
                '| [B](errors/b.md) | OLD | COLD | N/A |\n'
                '| [C](errors/measured-only.md) | OLD | COLD | N/A |\n')
        rows = [dict(shape=[9], chip=chip, data_role=role, roofline={'utilization': u}, cold_u=cold)
                for chip, role, u, cold in [(DT, 'replay', 1.5, 1), (DT, 'independent', .125, .125),
                                            (DT, 'independent', None, None), (PR, 'replay', 0, 0)]]
        rendered = summary_ranges(header + body, {'a': rows, 'b': []})
        self.assertEqual(rendered, header + body.replace('OLD', '0.125–1.5× / 0–0×', 1)
                         .replace('COLD', '12.500%-100.000% / 0.000%-0.000%', 1)
                         .replace('COLD', 'N/A / N/A').replace('OLD', 'N/A / N/A'))

    def test_duplicate_geometry_preserves_history_and_trace_rows(self):
        price = {'ideal_us': 1, 'bound': 'HBM', 'utilization': .25, 'reason': None}
        base = dict(op='mhc_post', shape=[8, 4, 256], input_dtype='bf16', attrs=None,
                    chip=DT, api='aclnnMhcPost', provider='ops-transformer', kernel=['MhcPost'],
                    ca_us=4, predicted_us=4, cold_us=4, cold_u=.25, signed_error_pct=0, query_ms=.1, query_970_ms=.1,
                    data_role='replay', extrapolated_970dt_us=3, roofline=price, future_roofline=price)
        rows = [dict(base, line=4, prefix=None), dict(base, line=5, prefix=[]),
                dict(base, line=6, prefix=[{'forget': True}])]
        page = render_contract('mhc_post · stream', rows)
        self.assertEqual(page.count('| 8×4×256 |'), 3)
        for line in (4, 5, 6):
            self.assertIn(f'windows.jsonl#L{line}', page)
        self.assertIn('未传 / 未传', page)
        self.assertIn('未传 / 空流', page)
        self.assertIn('[H1](#h1)', page)
        self.assertIn('{"forget":true}', page)

    def test_cold_io_and_original_are_not_scored_as_same_contract(self):
        case = dict(op='cast_bf16_fp32', shape=[262144, 64], attrs=None, dtype=None, prefix=None)
        original, _, logical, physical, cold = reprice(encoded(case), PR)
        direct = estimate(case['op'], case['shape'], chip=PR, memory_policy='cold_io')
        self.assertIsNone(physical['utilization'])
        self.assertGreater(logical['ideal_us'] / original, 2.8)
        self.assertAlmostEqual(cold['duration_us'], direct.latency_us)
        self.assertAlmostEqual(logical['ideal_us'] / cold['duration_us'], 1)

    def test_missing_cube_peak_still_has_cold_scenario_latency(self):
        case = dict(op='mhc_pre', shape=[1, 4, 512], attrs=None, dtype=None, prefix=None)
        original, _, logical, _, cold = reprice(encoded(case), PR)
        self.assertIsNone(logical['ideal_us'])
        self.assertGreaterEqual(cold['duration_us'], original)
