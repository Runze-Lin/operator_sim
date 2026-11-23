import unittest
from dataclasses import replace

from operator_model import estimate, load_chip
from operator_model.operators.vector.cache_load_antimx import core_grid


class AntiMxCacheTests(unittest.TestCase):
    def test_capture_layout_and_unbound_contract(self):
        shape=(1,128,512)
        attrs={'implementation':'aclnnAntiMxQuant','scale_shape':[1,128,8,2]}
        for sku,cores in [('Ascend950DT_9582',64),('Ascend950PR_957b',43)]:
            p=estimate('cache_load',shape,attrs,dtype='mxfp4',chip=sku)
            self.assertEqual(p.status,'estimated',p.reason)
            self.assertEqual(p.active_aiv,cores)
            self.assertEqual(p.work['vec_ops_by_dtype'],{'bf16':65536})
            self.assertEqual(p.kernel_count,1)
        self.assertIsNone(estimate('cache_load',shape,dtype='mxfp4').latency_us)
        self.assertIsNone(estimate('cache_load',(1,129,512),attrs={'implementation':'aclnnAntiMxQuant'},
                                   dtype='mxfp4',timing_mode='calibrated').latency_us)

    def test_grid_and_large_scaling(self):
        self.assertEqual(core_grid(9,576,64),(9,2,1,512))
        attrs={'implementation':'aclnnAntiMxQuant'}
        chip=load_chip('Ascend970DT')
        p=estimate('cache_load',(64,131081,2,16,64),attrs,dtype='mxfp4',chip=chip)
        self.assertEqual(p.status,'extrapolated',p.reason)
        self.assertGreater(p.projection['target_plan']['ub_tiles'],100)
        slower=replace(chip,hbm_bandwidth_gbs=100)
        self.assertGreater(estimate('cache_load',(64,131081,2,16,64),attrs,dtype='mxfp4',chip=slower).latency_us,
                           10*p.latency_us)

    def test_guards_do_not_reinterpret_the_packing(self):
        for dtype,shape in [('fp8',(1,64,128)),('mxfp4',(3,65)),('mxfp4',(3,0,128))]:
            p=estimate('cache_load',shape,{'implementation':'aclnnAntiMxQuant'},dtype=dtype)
            self.assertIsNone(p.latency_us)
