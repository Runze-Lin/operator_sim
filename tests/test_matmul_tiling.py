from dataclasses import replace
from unittest import TestCase
from operator_model.operators.matmul.tiling import matmul_plan
from operator_model.chips import load_chip


class TestMatmulRebalance(TestCase):
    def test_protocol_matches_archived_mmad_grids(self):
        pr=load_chip('Ascend950PR_957b')
        for shape,tile,cores,mmads in [
            ((128,512,512),[48,64,256],24,48),((256,512,256),[96,64,128],24,48),
            ((512,512,256),[80,128,128],28,56),((65,128,129),[16,64,144],10,10)]:
            p=matmul_plan(shape,pr)
            self.assertEqual((p['tile'],p['active_aic'],p['mmads_total']),(tile,cores,mmads))

    def test_scenario_uses_resources_not_sku_name(self):
        pr=load_chip('Ascend950PR_957b')
        sh=(256,512,256)
        anonymous=replace(pr,name='hypothetical')
        self.assertEqual(matmul_plan(sh,pr),matmul_plan(sh,anonymous))
        small=replace(anonymous,details={**pr.details,'l0a_bytes':32*1024,'l0b_bytes':32*1024})
        p=matmul_plan(sh,small)
        self.assertLess(p['tile'][2],matmul_plan(sh,pr)['tile'][2])
        self.assertEqual(p['read_page_k']%p['tile'][2],0)
