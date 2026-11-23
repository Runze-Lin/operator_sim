import unittest
from collections import Counter

from operator_model.operators.quant.token_fp8 import coefficients
from calibration.__main__ import HERE
from calibration.diagnostics import rows
from calibration.system_reference import archive, campaign_rows, check_origin
from calibration.token_quant.fit import fit


class NativeEvidenceTest(unittest.TestCase):
    def test_token_quant_prices_rebuild_from_development_stages(self):
        dev=archive(HERE/'token_quant/development.json.gz')
        observations=[(o['receipt']['case'],o['terminal_phases']) for o in dev['observations']]
        fitted=fit(observations)
        self.assertEqual(fitted.keys(),coefficients().keys())
        for name,price in fitted.items():
            self.assertAlmostEqual(price,coefficients()[name],places=10)

    def test_each_new_shape_has_both_skus_and_measured_only_has_no_price(self):
        campaign=list(campaign_rows())
        check_origin(campaign)
        pairs=Counter((r['contract'],tuple(r['shape'])) for r in campaign if r['data_role']=='independent')
        self.assertEqual(len(pairs),40)
        self.assertEqual(set(pairs.values()),{2})
        evidence=list(rows())
        self.assertEqual(len(evidence),8)
        self.assertTrue(all(r['ca_us']>0 and r['predicted_us'] is None and
                            r['extrapolated_970dt_us'] is None for r in evidence))
