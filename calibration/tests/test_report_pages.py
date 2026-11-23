"""Regressions for reporting repeated, stateful windows and missing Roofline evidence."""
import unittest

from calibration.report_pages import DT, PR, render_contract, summary_ranges, utilization


class ErrorPageTests(unittest.TestCase):
    def test_main_summary_spans_all_comparable_windows_on_each_chip(self):
        header = '| 算子与输入合同（逐窗） | 等效 Roofline U 范围（DT / PR） | WAPE |\n|---|---|---|\n'
        body = ('| [A](errors/a.md) | OLD | 2% |\n'
                '| [B](errors/b.md) | OLD | N/A |\n'
                '| [C](errors/measured-only.md) | OLD | N/A |\n')
        rows = [dict(shape=[9], chip=chip, data_role=role, roofline={'utilization': u})
                for chip, role, u in [(DT, 'replay', 1.5), (DT, 'independent', .125),
                                      (DT, 'independent', None), (PR, 'replay', 0)]]
        rendered = summary_ranges(header + body, {'a': rows, 'b': []})
        self.assertEqual(rendered, header + body.replace('OLD', '12.500%-150.000% / 0.000%-0.000%', 1)
                         .replace('OLD', 'N/A / N/A'))

    def test_duplicate_geometry_preserves_history_and_trace_rows(self):
        price = {'ideal_us': 1, 'bound': 'HBM', 'utilization': .25, 'reason': None}
        base = dict(op='mhc_post', shape=[8, 4, 256], input_dtype='bf16', attrs=None,
                    chip=DT, api='aclnnMhcPost', provider='ops-transformer', kernel=['MhcPost'],
                    ca_us=4, predicted_us=4, signed_error_pct=0, query_ms=.1, query_970_ms=.1,
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

    def test_weighted_utilization_uses_the_same_eligible_model_windows(self):
        rows = [dict(predicted_us=2, roofline=dict(ideal_us=3, utilization=1.5)),
                dict(predicted_us=8, roofline=dict(ideal_us=1, utilization=.125)),
                dict(predicted_us=50, roofline=dict(ideal_us=None, utilization=None))]
        self.assertEqual(utilization(rows), (2, .4))
        self.assertEqual(utilization(rows[-1:]), (0, None))
