import json
from dataclasses import replace
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from operator_model import estimate, load_chip


class FAPublicResourceRoute(TestCase):

    def test_reference_resources_are_portable_and_long_kv_is_marked(self):
        chip = load_chip()
        shape = (9, 1008, 64)
        reference = estimate('fa',shape,chip=chip)
        anonymous = estimate('fa',shape,chip=replace(chip,name='scenario'))
        self.assertEqual(anonymous.latency_us, reference.latency_us)
        self.assertEqual(anonymous.projection['target_plan'], reference.projection['target_plan'])
        self.assertEqual(reference.status,'estimated')
        self.assertEqual(estimate('fa',(9,262144,64)).status,'extrapolated')
        self.assertEqual(estimate('fa',shape,{'causal':True}).status,'unsupported')
