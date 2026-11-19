from dataclasses import replace
import unittest

from operator_model import estimate, load_chip


class RmsResourceTest(unittest.TestCase):
    def test_uneven_partition_and_write_geometry(self):
        for name, active, rows in [('Ascend950DT_9582',64,1),('Ascend950PR_957b',32,2),('Ascend970DT',64,1)]:
            r=estimate('rmsnorm',(64,2048),chip=name,timing_mode='resource')
            d=r.projection['timing_diagnostics']
            self.assertEqual(r.active_aiv,active)
            self.assertEqual(d['target_plan']['rows_per_core'],rows)
            self.assertEqual(d['target_plan']['vector_groups'],16*rows)
            self.assertGreater(d['store_union_cycles'],d['store_eligible_overlap_cycles'])
            self.assertAlmostEqual(sum(r.projection['stages_us'].values()),r.latency_us)
            conditions=list(d['conditional_us'].values())
            self.assertAlmostEqual(r.latency_us,2*min(conditions)*max(conditions)/(min(conditions)+max(conditions)))

    def test_resource_sensitivity_and_ub(self):
        chip=load_chip()
        def get(c):return estimate('rmsnorm',(512,4096),{'resident_aiv':0},chip=c,timing_mode='resource')
        base=get(chip)
        self.assertEqual(base.projection['target_plan']['ub_tile_rows'],[7,1])
        for extra in ({'ub_bytes':1000},{'mte2_bandwidth_gbs':1},{'mte3_bandwidth_gbs':1}):
            r=get(replace(chip,details={**chip.details,**extra}))
            if 'ub_bytes' in extra:self.assertEqual(r.status,'unsupported')
            else:self.assertGreater(r.latency_us,base.latency_us)
        self.assertGreater(get(replace(chip,hbm_bandwidth_gbs=10)).latency_us,base.latency_us)
        self.assertEqual(get(replace(chip,name='scenario')).latency_us,base.latency_us)

    def test_context_contract_and_clock(self):
        chip=load_chip('Ascend950PR_957b')
        for h,expected in ((0,'first'),(5,'expand'),(32,'reuse')):
            r=estimate('rmsnorm',(64,2048),{'resident_aiv':h},chip=chip,timing_mode='resource')
            self.assertEqual(r.projection['timing_diagnostics']['residency'],expected)
        for bad in (-1,57,True):
            r=estimate('rmsnorm',(64,2048),{'resident_aiv':bad},chip=chip,timing_mode='resource')
            self.assertEqual(r.status,'unsupported')
        self.assertEqual(estimate('rmsnorm',(64,2048),{'new_layout':1},chip=chip).status,'unsupported')
        self.assertEqual(estimate('rmsnorm',(64,2048),chip=chip,timing_mode='calibrated').status,'unsupported')
