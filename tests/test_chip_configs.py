import json
from pathlib import Path
import unittest

from operator_model.chips import load_chip, REFERENCE_NAME


class ChipTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1]/'configs/chips'/(REFERENCE_NAME+'.json')
        self.doc = json.loads(self.path.read_text())

    def test_builtin_and_file_match(self):
        dt = load_chip()
        pr = load_chip('Ascend950PR_957b')
        self.assertEqual(load_chip(self.path), dt)
        self.assertIs(load_chip(dt), dt)
        self.assertEqual((dt.cube_cores, dt.hbm_capacity_gb*1e9), (32, 96*2**30))
        self.assertEqual((pr.vector_cores, pr.hbm_bandwidth_gbs), (56, 1400))
        self.assertLess(pr.cube_tops['16'], dt.cube_tops['16'])
        self.assertIsNone(pr.cube_tops['32'])

    def test_inheritance_and_snapshot(self):
        self.doc['name'] = 'future-scenario'
        self.doc['details'] = {'ub_bytes': 262144, 'l2_bandwidth_gbs': None}
        scenario = load_chip(self.doc)
        self.assertEqual(scenario.details['l2_bytes'], load_chip().details['l2_bytes'])
        self.assertIn('details.l2_bytes', scenario.inherited_fields)
        self.assertNotIn('details.ub_bytes', scenario.inherited_fields)
        self.doc['details']['ub_bytes'] = 1
        self.assertEqual(scenario.details['ub_bytes'], 262144)
        with self.assertRaises(TypeError):
            scenario.cube_tops['16'] = 1

    def test_reject_bad_units_and_unknown_fields(self):
        for field, value in [('cube_cores', 2.1), ('cube_tops', {'16': 1}),
                             ('hbm_capacity_gb', 0), ('frequency_ghz', float('nan')),
                             ('details', {'mte_bandwith': 2}), ('schema_version', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_chip({**self.doc, field:value})
