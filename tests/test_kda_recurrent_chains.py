"""Independent checks of the A5 host partition and exposed-state assumptions."""
from dataclasses import replace
from unittest import TestCase
from unittest.mock import patch

from operator_model.chips import load_chip
from operator_model.operators.kda.recurrent_chains import estimate_recurrent_chains
from operator_model.backend import Estimate


class RecurrentChainsTest(TestCase):
    shape=(128,1,12,12,128,128)

    def test_kimi_geometry_and_payload(self):
        dt=estimate_recurrent_chains(self.shape)
        pr=estimate_recurrent_chains(self.shape,chip='Ascend950PR_957b')
        self.assertEqual((dt['chains'],dt['active_aiv'],dt['chains_per_busiest_aiv'],dt['cores_with_full_rounds']), (1536,64,24,64))
        self.assertEqual((pr['active_aiv'],pr['chains_per_busiest_aiv'],pr['cores_with_full_rounds']), (56,28,24))
        self.assertEqual((dt['tiling']['v_step'],dt['tiling']['stripes']), (64,2))
        state_bytes=1536*128*128*4
        self.assertEqual(dt['unique_read_bytes'],state_bytes+1536*(3*2*128+4*128+4))
        self.assertEqual(dt['unique_write_bytes'],state_bytes+1536*2*128)
        self.assertLess(dt['latency_us'],dt['repeated_cold_windows_us'])
        self.assertFalse(dt['calibrated_multichain'])
        longer=estimate_recurrent_chains([128,3,12,12,128,128]);self.assertEqual(longer['logical_state_store_descriptor_bytes'],3*state_bytes)
        self.assertEqual(longer['unique_state_bytes'],state_bytes)

    def test_serial_depth_and_cold_hbm_are_different_constraints(self):
        one=estimate_recurrent_chains([1,1,1,1,128,128]);many=estimate_recurrent_chains([1,1,12,12,128,128]);self.assertAlmostEqual(one['latency_us'],many['latency_us'])
        cold=estimate_recurrent_chains(self.shape,hbm_efficiency=.01)
        warm=estimate_recurrent_chains(self.shape,hbm_efficiency=.01,hbm_read_fraction=0,hbm_write_fraction=0)
        self.assertAlmostEqual(cold['latency_us'],cold['cold_io_floor_us'])
        self.assertEqual(warm['external_service_floor_us'],0)
        self.assertLess(warm['latency_us'],cold['latency_us']);self.assertGreater(warm['latency_us'],0)

    def test_different_contract_or_ub_does_not_inherit_a_time(self):
        for dtype in ('fp16','int4'):
            with self.assertRaises(ValueError):estimate_recurrent_chains(self.shape,{'dtype':dtype})
        with self.assertRaises(ValueError):estimate_recurrent_chains([3,9,12,12,128,128])
        with self.assertRaises(ValueError):estimate_recurrent_chains(self.shape,{'state_v_first':True})
        for ub in (None,8192,1024*1024):
            changed=replace(load_chip(),details={'ub_bytes':ub})
            trial=estimate_recurrent_chains(self.shape,chip=changed)
            self.assertEqual(trial['status'],'unpriced_tiling');self.assertIsNone(trial['latency_us']);self.assertEqual(trial['chains'],1536)
        with patch('operator_model.backend.estimate',return_value=Estimate(.01,'estimated')):
            self.assertEqual(estimate_recurrent_chains(self.shape)['status'],'incompatible_stage_prices')

    def test_forward_original_host_attrs_only_for_the_same_dispatch(self):
        a=dict(layout='BSND',dtype='BF16',gate_dtype='FP32',state_dtype='FP32',use_gate_in_kernel=False,state_v_first=False,chunk_size=None,output_final_state=True,export_gk=False)
        self.assertEqual(estimate_recurrent_chains(self.shape,a)['latency_us'],estimate_recurrent_chains(self.shape)['latency_us'])
        for k,v in [('export_gk',True),('chunk_size',64),('normalized',False),('cu_seqlens',[0,1]),('inplace_final_state',True),('use_gate_in_kernel',1),('scale',1.)]:
            with self.subTest(k=k),self.assertRaises(ValueError):estimate_recurrent_chains(self.shape,{**a,k:v})
        with patch('operator_model.backend.estimate',return_value=Estimate(None,'unsupported','no projected price')):
            self.assertEqual(estimate_recurrent_chains(self.shape)['status'],'unpriced_single_chain')
