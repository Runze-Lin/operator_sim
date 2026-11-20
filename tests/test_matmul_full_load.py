"""Candidate-only B-resident geometry, resource units and branch boundaries."""
from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.matmul_full_load import estimate_matmul_full_load


class FullLoadTests(TestCase):
    def test_b_is_read_once_per_aic_and_output_is_bf16(self):
        row=estimate_matmul_full_load([262144,128,512])
        self.assertEqual(row['tile'],(256,128,32))
        self.assertEqual(row['a_page_k'],128)
        self.assertEqual(row['waves'],32)
        self.assertEqual(row['mmad_total'],32*32*16)
        self.assertEqual(row['work']['b_lower_link_bytes'],32*512*128*2)
        self.assertEqual(row['work']['a_read_bytes'],262144*512*2)
        self.assertEqual(row['work']['final_bf16_store_bytes'],262144*128*2)
        self.assertEqual(row['work']['fixp_fp32_source_bytes'],262144*128*4)
        self.assertEqual(row['cycles']['cube_issue'],257)
        self.assertEqual(row['cycles']['per_aic_step'],16*257)

    def test_global_hbm_and_configuration_change_the_bottleneck(self):
        short=[32768,128,512]
        pr=estimate_matmul_full_load(short,chip='Ascend950PR_957b')
        self.assertEqual(pr['active_aic'],28)
        self.assertEqual(pr['waves'],5)
        self.assertEqual(pr['bottleneck'],'hbm')
        dt=load_chip()
        slower=replace(dt,name='limited-memory-scenario',hbm_bandwidth_gbs=1000)
        limited=estimate_matmul_full_load([262144,128,512],chip=slower)
        base=estimate_matmul_full_load([262144,128,512],chip=dt)
        self.assertTrue(limited['target_extrapolated'])
        self.assertEqual(limited['bottleneck'],'hbm')
        self.assertGreater(limited['latency_us'],base['latency_us'])
        self.assertLessEqual(base['latency_us'],base['all_output_writeback_us'])
        self.assertEqual(limited['rates']['hbm_bytes_per_chip_cycle'],1000/dt.frequency_ghz)

    def test_unpriced_branches_return_no_latency(self):
        for kwargs in ({'dtype':'fp32'},{'layout':'NZ_ND'}):
            self.assertIsNone(estimate_matmul_full_load([262144,128,512],**kwargs)['latency_us'])
        self.assertIsNone(estimate_matmul_full_load([32768,128,512],chip='Ascend970DT')['latency_us'])
        self.assertIsNone(estimate_matmul_full_load([262145,128,512])['latency_us'])
        with self.assertRaises(ValueError):
            estimate_matmul_full_load([True,128,512])
