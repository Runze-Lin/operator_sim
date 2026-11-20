from dataclasses import replace
from unittest import TestCase
from operator_model import load_chip
from operator_model.operators.matmul.placement import matmul_placement


class MatmulPlacementTests(TestCase):
    def test_split_changes_with_target_count_and_k_alignment(self):
        dt=load_chip('Ascend950DT_9582');pr=load_chip('Ascend950PR_957b')
        for target, edge in ((dt,4096),(pr,3584)):
            for m in (edge,edge+1):
                r=matmul_placement([m,32,5120],chip=target)
                self.assertEqual(r['route'], 'split_k' if m==edge else 'ordinary')
            self.assertEqual(matmul_placement([17,64,3840],chip=target)['route'],'ordinary')
            self.assertEqual(matmul_placement([17,64,3841],chip=target)['route'],'split_k')
            self.assertEqual(matmul_placement([17,64,3841],chip=target),
                             matmul_placement([17,64,3841],chip=replace(target,name='other')))

    def test_per_core_work_reduced_at_the_real_asymmetric_boundary(self):
        dt=matmul_placement([4096,32,5120]);pr=matmul_placement([4096,32,5120],chip='Ascend950PR_957b')
        self.assertEqual((dt['host_groups'],dt['splits_per_output'],dt['max_segment_k']),(16,2,2560))
        self.assertEqual(dt['fp32_reduction_adds'],4096*32)
        self.assertEqual(pr['route'],'ordinary')
        self.assertGreater(dt['latency_us'],0)
        self.assertEqual(matmul_placement([144,64,7168],chip='Ascend970DT')['route'],'ordinary')

    def test_invalid_input_and_tile_capacity(self):
        with self.assertRaises(ValueError):matmul_placement([0,16,4096])
        dt=load_chip('Ascend950DT_9582')
        smaller=replace(dt,details={**dt.details,'l0a_bytes':32768})
        self.assertEqual(matmul_placement([3584,32,5120],chip=smaller)['host_groups'],28)
