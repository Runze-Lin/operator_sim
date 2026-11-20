from dataclasses import replace
import json
from pathlib import Path
import unittest

from operator_model import estimate, load_chip


class CastResourceTest(unittest.TestCase):
    def test_two_slot_predictions_match_the_frozen_candidate(self):
        archive = Path(__file__).resolve().parents[1] / 'results/cast-capacity'
        frozen = json.loads((archive / 'two-slot-freeze.json').read_text())
        retention = json.loads((archive.parent/'v0.25/cache/replay.json').read_text())
        price = {(r['chip'],tuple(r['shape'])):r['forecasts']['pressure']['us']
                 for r in retention['rows'] if r['op']=='cast_bf16_fp32'}
        for job in frozen['included'] + frozen['excluded']:
            for sku, target in job['targets'].items():
                with self.subTest(shape=job['shape'], chip=sku):
                    chip = load_chip(sku)
                    prediction = estimate('cast_bf16_fp32', job['shape'], chip=chip)
                    d = prediction.projection['timing_diagnostics']
                    self.assertAlmostEqual(prediction.latency_us, price.get((sku,tuple(job['shape'])),target['candidate_us']))
                    self.assertEqual(prediction.active_aiv, target['active_aiv'])
                    self.assertEqual(d['target_plan']['busy_core_waves'], target['waves'])
                    slot_pair = (d['input_slot_ii_cycles'], d['output_slot_ii_cycles'])
                    if target['slot_floor_us'] is None:
                        self.assertEqual(slot_pair, (None, None))
                    else:
                        self.assertAlmostEqual(max(slot_pair)/(1000*chip.frequency_ghz),
                                               target['slot_floor_us'])
        replay = json.loads((archive / 'replay.json').read_text())
        for row in replay['rows']:
            with self.subTest(shape=row['shape'], chip=row['chip']):
                p = estimate('cast_bf16_fp32', row['shape'], chip=row['chip'])
                self.assertAlmostEqual(p.latency_us, price[(row['chip'],tuple(row['shape']))])

    def test_independent_slot_reuse_and_target_resources(self):
        chip = load_chip()

        def query(frequency=1.65, read=300, write=300, hbm=10000):
            target = replace(chip, name='custom', frequency_ghz=frequency, hbm_bandwidth_gbs=hbm,
                             details={**chip.details, 'mte2_bandwidth_gbs': read,
                                      'mte3_bandwidth_gbs': write, 'l2_bandwidth_gbs': 50000})
            return estimate('cast_bf16_fp32', (262144,32), chip=target)

        fast, slow_read, slow_write = query(), query(read=5), query(write=5)
        input_key, output_key = 'input_slot_ii_cycles', 'output_slot_ii_cycles'
        base, rd, wr = [r.projection['timing_diagnostics'] for r in (fast, slow_read, slow_write)]
        self.assertGreater(rd[input_key], base[input_key])
        self.assertEqual(rd[output_key], base[output_key])
        self.assertGreater(wr[output_key], base[output_key])
        self.assertEqual(wr[input_key], base[input_key])
        for slow in (slow_read, slow_write, query(frequency=.825), query(hbm=100)):
            self.assertGreater(slow.latency_us, fast.latency_us)

    def test_exact_conversion_work_and_flattened_partition(self):
        chip=load_chip()
        left=estimate('cast_bf16_fp32',(64,4096),chip=chip)
        right=estimate('cast_bf16_fp32',(128,2048),chip=chip)
        self.assertEqual(left.latency_us,right.latency_us)
        self.assertEqual(left.active_aiv,chip.vector_cores)
        self.assertEqual(left.work['logical_write_bytes'],4*64*4096)
        self.assertEqual(left.work['logical_read_bytes'],2*64*4096)
        self.assertEqual(left.work['vec_ops_by_class']['bf16_to_fp32_conversion'],64*4096)
        self.assertEqual(left.work['cube_ops'],0)
        self.assertAlmostEqual(sum(left.projection['stages_us'].values()), left.latency_us)

    def test_resource_limits_and_contract(self):
        chip=load_chip()
        query=lambda c:estimate('cast_bf16_fp32',(262144,7168),chip=c,timing_mode='resource')
        base=query(chip)
        slower=query(replace(chip,hbm_bandwidth_gbs=chip.hbm_bandwidth_gbs/2))
        self.assertGreater(slower.latency_us,base.latency_us)
        self.assertEqual(query(replace(chip,name='renamed')).latency_us,base.latency_us)
        self.assertEqual(estimate('cast_bf16_fp32',(5,320),dtype='fp16').status,'unsupported')
        self.assertEqual(estimate('cast_bf16_fp32',(1,0)).status,'unsupported')

    def test_shared_read_write_resources_follow_residency_and_spill(self):
        chip=load_chip('Ascend950DT_9582')
        def query(shape, **overrides):
            target=replace(chip,details={**chip.details,**overrides})
            return estimate('cast_bf16_fp32',shape,chip=target,timing_mode='resource')
        fits=query((262144,32))
        d=fits.projection['timing_diagnostics']
        self.assertEqual(d['modeled_external_bytes'],fits.work['logical_read_bytes'])
        self.assertEqual(d['assumed_retained_write_bytes'],fits.work['logical_write_bytes'])
        spilled=query((2097152,32)).projection['timing_diagnostics']
        self.assertGreater(spilled['modeled_external_write_bytes'],0)
        self.assertEqual(spilled['assumed_retained_write_bytes']+spilled['l2_input_reserve_bytes'],
                         chip.details['l2_bytes'])
        no_capacity=query((262144,32),l2_bytes=None)
        self.assertEqual(no_capacity.projection['timing_diagnostics']['modeled_external_bytes'],6*262144*32)
        slow_l2=query((262144,32),l2_bandwidth_gbs=100)
        self.assertGreater(slow_l2.latency_us,fits.latency_us)
        self.assertAlmostEqual(slow_l2.latency_us,(1753+6*262144*32*chip.frequency_ghz/100)/(1000*chip.frequency_ghz))
