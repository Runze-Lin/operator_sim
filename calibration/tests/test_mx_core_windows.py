"""MX AIV retirement can outlast scalar END and the merged trace window."""

import json
import tempfile
import unittest
from pathlib import Path

from calibration.mx_matmul.collect import core_windows


class MxPhysicalWindowTests(unittest.TestCase):
    def test_sync_retirement_and_roster(self):
        with tempfile.TemporaryDirectory() as folder:
            simulator = Path(folder)
            core = simulator/'core0.veccore0'
            core.mkdir()
            events = [
                dict(ph='X', name='MOV_XD_SPR', ts=1.0, dur=.02),
                dict(ph='X', name='WAIT_INTRA_BLOCK', ts=1.1, dur=.9),
                dict(ph='X', name='END', ts=1.3, dur=0),
            ]
            shard = core/'trace.json'
            shard.write_text(json.dumps(dict(traceEvents=events)))
            roster = {core.name:1.0}
            cores, hashes = core_windows(simulator, roster)
            self.assertEqual(cores[core.name], dict(first_tick=1650, end_tick=2145, retire_tick=3300, end_count=1))
            self.assertEqual(len(hashes[core.name]), 64)
            with self.assertRaisesRegex(ValueError, 'roster'):
                core_windows(simulator, {**roster, 'core0.veccore1': 1.0})
            with self.assertRaisesRegex(ValueError, 'duration mismatch'):
                core_windows(simulator, {core.name:.3})
            shard.write_text(json.dumps(dict(traceEvents=events[:-1])))
            with self.assertRaisesRegex(ValueError, 'Physical END'):
                core_windows(simulator, roster)


if __name__ == '__main__':
    unittest.main()
