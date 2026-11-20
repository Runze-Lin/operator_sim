from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.operators.attention.residual_work import attention_residual_work
from operator_model.operators.attention.residual_timing import resource_attention


class KimiAttentionResidual(TestCase):
    def test_dispatch_and_reload(self):
        r=attention_residual_work(9,7168,8,6,addend=True,materialize=True,write_idx=6)
        self.assertEqual((r['max_resident_rows'],r['tiling_key'],r['payload_aiv']), (4,40010,9))
        self.assertEqual((r['dma_read_bytes'],r['dma_write_bytes']),(2193408,516096))
        # 65 tokens, 64 launches with ceildiv(65,64)=2 contiguous tokens per core.
        p=attention_residual_work(65,512,8,2,optimize_prefill=True)
        self.assertEqual((p['tiling_key'],p['launch_aiv'],p['payload_aiv']),(40021,64,33))
        self.assertEqual(attention_residual_work(0,512,8,0)['dma_read_bytes'],0)

    def test_pricing_requires_measured_branch(self):
        ch=load_chip()
        for attrib in ({},{'materialize':True,'write_idx':1}):
            t,w,d=resource_attention([1,7168,8,1],ch,attrib)
            self.assertIsNone(t)
            self.assertGreater(w['vec_ops'],0)
        a={'materialize':True,'write_idx':1,'addend':True}
        baseline,w,d=resource_attention([1,7168,8,1],ch,a)
        self.assertGreater(baseline,0)
        self.assertAlmostEqual(sum(d['stages_us'].values()),baseline)
        target=replace(ch,name='renamed',vector_tops={**ch.vector_tops,'32':9})
        self.assertGreater(resource_attention([1,7168,8,1],target,a)[0],baseline)
        self.assertIsNone(resource_attention([1,7168,8,6],ch,{**a,'write_idx':6})[0])
        with self.assertRaises(ValueError):
            resource_attention([3,64,8,2],ch,{'write_idx':1})
