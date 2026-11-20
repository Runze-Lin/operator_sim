from dataclasses import replace
from unittest import TestCase
from operator_model import estimate,load_chip

ATTRS=dict(addend=True, write_idx=-1,materialize=False,output_norm=True,mix=True,optimize_prefill=True)

class AttnResSibling(TestCase):
    def test_branch_query_and_public_work(self):
        for sku,launch,payload,rounds in [('Ascend950DT_9582',64,64,8),('Ascend950PR_957b',56,52,10),('Ascend970DT',120,103,5)]:
            r=estimate('attn_res_fwd',[512,7168,8,1],ATTRS,chip=sku)
            self.assertEqual(r.status,'extrapolated')
            self.assertEqual(r.active_aiv,payload)
            d=r.projection['timing_diagnostics']
            self.assertEqual((d['launch_aiv'],d['tokens_per_core'],d['tiling_key']),(launch,rounds,40021))
            self.assertEqual(d['dma_read_bytes'],14336*(3*512+3*payload))
            self.assertEqual(r.work['logical_write_bytes'],2*512*14336)
            self.assertAlmostEqual(sum(r.projection['stages_us'].values()),r.latency_us)
            cal=estimate('attn_res_fwd',[512,7168,8,1],ATTRS,chip=sku,timing_mode='calibrated')
            self.assertEqual(cal.status,'unavailable');self.assertIsNone(cal.latency_us)

    def test_existing_contracts_and_dispatch_boundary(self):
        c='Ascend950DT_9582'
        def result(t,n,**over):
            return estimate('attn_res_fwd',[t,7168,8,n],{**ATTRS,**over},chip=c)
        a,b=result(64,1),result(65,1)
        self.assertEqual(a.projection['target_plan']['tiling_key'],40020)
        self.assertEqual(b.projection['target_plan']['tiling_key'],40021)
        self.assertEqual((a.active_aiv,b.active_aiv),(64,33))
        self.assertGreater(b.latency_us,a.latency_us)
        self.assertIsNone(result(65,7).latency_us) # the resident UB branch cannot fit
        self.assertIsNone(result(65,1,materialize=True).latency_us)
        self.assertIsNone(result(65,1,write_idx=1).latency_us)
        self.assertIsNone(result(65,0).latency_us)
        self.assertIsNotNone(result(1,1,materialize=True,write_idx=1).latency_us)

    def test_resource_changes_and_many_rows(self):
        c=load_chip('Ascend950DT_9582');s=[262144,7168,8,1]
        base=estimate('attn_res_fwd',s,ATTRS,chip=c)
        self.assertGreater(base.latency_us,0)
        slow=replace(c,name='low vector',vector_tops={**c.vector_tops,'32':c.vector_tops['32']/2})
        self.assertGreater(estimate('attn_res_fwd',s,ATTRS,chip=slow).latency_us,base.latency_us)
        limited=replace(c,name='low HBM',hbm_bandwidth_gbs=8)
        x=estimate('attn_res_fwd',s,ATTRS,chip=limited)
        self.assertGreater(x.projection['stages_us']['exposed_shared'],0)
        self.assertAlmostEqual(x.latency_us,x.work['logical_read_bytes']/8000)
        mte=replace(c,name='slow MTE',details={**c.details,'mte2_bandwidth_gbs':1.})
        self.assertGreater(estimate('attn_res_fwd',s,ATTRS,chip=mte).latency_us,base.latency_us)
        missing=replace(c,name='unknown throughput',vector_tops={**c.vector_tops,'32':None})
        self.assertIsNone(estimate('attn_res_fwd',s,ATTRS,chip=missing).latency_us)
