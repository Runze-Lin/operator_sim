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
            self.assertEqual(d['residency_assumption'],'unknown_stream_history')
            p = (active-1)/load_chip(name).vector_cores
            states = d['conditional_us']
            running = states['reuse']*(1-p) + states['expand']*p
            self.assertAlmostEqual(r.latency_us,2*running*states['first']/(running+states['first']))

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

    def test_long_prefill_plan_is_run_length_encoded(self):
        chip = load_chip()
        r = estimate('rmsnorm', (8*32768, 7168), chip=chip, timing_mode='resource')
        plan = r.projection['target_plan']
        self.assertEqual(plan['iterations'], 1024)
        self.assertEqual(plan['full_tile_count']*plan['full_tile_rows']+plan['tail_tile_rows'], 4096)
        self.assertNotIn('ub_tile_rows',plan)
        self.assertGreaterEqual(r.latency_us*1e3*chip.hbm_bandwidth_gbs,
                                r.projection['timing_diagnostics']['modeled_external_bytes']*(1-1e-12))
        self.assertEqual(estimate('rmsnorm',(262144,16),chip=chip,timing_mode='resource').status,'unsupported')

    def test_write_retirement_depends_on_capacity_and_bandwidth(self):
        chip = load_chip('Ascend950PR_957b')
        def diag(shape, **hardware):
            target = replace(chip, details={**chip.details, **hardware})
            return estimate('rmsnorm', shape, chip=target).projection['timing_diagnostics']
        fits = diag((2048, 5120))
        self.assertEqual(fits['assumed_retained_write_bytes'], fits['write_bytes'])
        for detail in ({'l2_bytes': None}, {'rms_output_retained_fraction': 0}):
            cold = diag((2048, 5120), **detail)
            self.assertEqual(cold['modeled_external_bytes'], cold['cold_read_bytes'] + cold['write_bytes'])
            self.assertGreater(cold['shared_io_floor_cycles'], fits['shared_io_floor_cycles'])
        spill = diag((262144, 5120), l2_bandwidth_gbs=100)
        self.assertLess(spill['assumed_retained_write_bytes'], spill['write_bytes'])
        self.assertEqual(spill['shared_io_floor_cycles'], spill['l2_floor_cycles'])
        with self.assertRaises(ValueError):
            replace(chip, details={**chip.details, 'rms_output_retained_fraction': 1.1})

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
