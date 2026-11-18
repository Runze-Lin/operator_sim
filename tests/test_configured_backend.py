import json
import subprocess
import sys
import unittest
from pathlib import Path

from operator_model import estimate, load_chip

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT/'configs/chips/scenario-example.json'


class ConfiguredBackendTest(unittest.TestCase):
    def test_custom_chip_via_path_and_loaded_object(self):
        chip = load_chip(EXAMPLE)
        large = estimate('rmsnorm', [512, 4096], dtype='bfloat16', chip=chip)
        path = estimate('rmsnorm', [512, 4096], {'dtype': 'BF16'}, chip=EXAMPLE)
        self.assertEqual(large, path)
        self.assertEqual(large.status, 'extrapolated')
        self.assertIn('details.ub_bytes', large.projection['inherited_fields'])
        self.assertEqual(large.work, estimate('rmsnorm', [512, 4096]).work)
        self.assertGreater(large.rates['counted_vector_tops'], 0)
        self.assertIsNone(large.work['physical_hbm_bytes'])

    def test_dtype_is_checked_even_for_frozen_routes(self):
        args = ('matmul', [32, 64, 128])
        self.assertEqual(estimate(*args, dtype='bfloat16'), estimate(*args))
        self.assertEqual(estimate(*args, dtype='FP16').status, 'unsupported')
        self.assertEqual(estimate(*args, {'dtype':'fp32'}, dtype='bf16').status, 'unsupported')
        self.assertEqual(estimate(*args, {'dtype':None}).status, 'unsupported')
        self.assertEqual(estimate('kda_chunk',[1,64,1,1,128,128],dtype='fp16').status, 'unavailable')

    def test_same_name_modified_peaks_do_not_claim_calibration(self):
        config = json.loads((EXAMPLE.parent/'Ascend950DT_9582.json').read_text())
        config['vector_tops']['32'] /= 2
        res = estimate('rmsnorm', [512, 4096], chip=config)
        self.assertEqual(res.status, 'extrapolated')
        self.assertGreater(res.latency_us, estimate('rmsnorm',[512,4096]).latency_us)

    def test_accounting_units_and_reference_occupancy(self):
        res = estimate('matmul', [16,64,128])
        self.assertEqual(res.active_aic, 1)
        self.assertAlmostEqual(res.rates['cube_nominal_peak_fraction'],res.rates['useful_cube_tops']/432)
        self.assertAlmostEqual(res.rates['useful_cube_tops']*res.latency_us*1e6,2*16*64*128)
        fa = estimate('fa',[9,512,64])
        self.assertEqual(fa, estimate('fa',[9,512,64,1,1]))
        self.assertIsNotNone(fa.active_aiv)

    def test_target_state_is_counted_against_target_cores(self):
        op, shape = 'rmsnorm', [128, 4096]
        self.assertEqual(estimate(op, shape, {'resident_aiv':121}, chip=EXAMPLE).status,'unsupported')
        warm = estimate(op, shape, {'resident_aiv':120}, chip=EXAMPLE)
        cold = estimate(op, shape, {'resident_aiv':0}, chip=EXAMPLE)
        self.assertEqual(cold.projection['resident_aiv_mapping']['reference'], 0)
        self.assertGreater(warm.projection['resident_aiv_mapping']['reference'], 0)
        self.assertNotEqual(warm.latency_us, cold.latency_us)

    def test_cli_exposes_same_work_and_target_status_as_api(self):
        out = subprocess.check_output([sys.executable,'-m','operator_model','--op','fa','--shape',
            '9','512','64','--dtype','bf16','--chip',str(EXAMPLE)], text=True)
        got = json.loads(out)
        ref = estimate('fa',[9,512,64],chip=EXAMPLE)
        self.assertEqual((got['soc_version'],got['estimate_status'],got['work']),
                         (ref.chip_name,ref.status,ref.work))
        self.assertEqual(got['predicted_us'],ref.latency_us)
