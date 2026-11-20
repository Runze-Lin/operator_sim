"""The streaming quant sensitivity uses configurable resources and whole-stream traffic."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from operator_model import estimate, load_chip
from operator_model.quant_stream_memory import scenario

ROOT=Path(__file__).resolve().parents[1]


class TestQuantStreamMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze=json.loads((ROOT/'results/v0.29/quant/freeze.json').read_bytes())

    def test_no_extra_mixed_bus_time_for_sorting_the_small_stream(self):
        for sku in ('Ascend950DT_9582','Ascend950PR_957b'):
            c=load_chip(sku)
            for shape in ([5,320],[512,4096],[2048,5120]):
                old=estimate('quant',shape,chip=c).latency_us
                new=scenario(shape,c,old,mixed_bus_efficiency=.6)
                self.assertEqual(new['latency_us'],old)
                self.assertEqual(new['mixed_us'],0.)
                self.assertEqual(new['dirty_retained_bytes'],new['logical_write'])

    def test_reproduce_new_shape_and_no_shape_id_switch(self):
        for key,shape in self.freeze['shapes'].items():
            for sku,v in self.freeze['predictions'][key].items():
                c=load_chip(sku);q=scenario(shape,c,v['base_us'],mixed_bus_efficiency=v['mixed_bus_efficiency'],
                                            wire_to_configured_bandwidth=v['wire_to_configured_bandwidth'])
                self.assertEqual(q['latency_us'],v['latency_us'])
                self.assertAlmostEqual(q['logical_write'],q['dirty_retained_bytes']+q['write_mixed_bytes'],delta=1.e-5)
                self.assertAlmostEqual(q['logical_read'],q['read_mixed_bytes']+q['read_prefix_bytes'],delta=1.e-5)

    def test_capacity_pressure_and_concurrent_issue_do_not_stack(self):
        c=load_chip();shape=[65536,7168]
        base=estimate('quant',shape,chip=c).latency_us
        def compute(cap,eta):
            chip=replace(c,details=dict(c.details)|{'l2_bytes':cap})
            return scenario(shape,chip,base,mixed_bus_efficiency=eta)
        small=compute(32*2**20,.75);large=compute(256*2**20,.75);ideal=compute(32*2**20,1.)
        self.assertGreater(small['write_mixed_bytes'],large['write_mixed_bytes'])
        self.assertGreater(small['latency_us'],ideal['latency_us'])
        self.assertGreater(small['latency_us'],large['latency_us'])
        self.assertEqual(small['latency_us'],max(base,small['prefix_us']+small['mixed_us'],*small['critical_local_us'].values()))
        with self.assertRaises(ValueError):
            compute(32*2**20,1.1)

    def test_consistent_capacity_crossing(self):
        c=load_chip();cols=7168;rows=4000;r=2*rows*cols;w=rows*(cols+2*(cols//64))
        base=estimate('quant',[rows,cols],chip=c).latency_us
        vals=[]
        for delta in (-2,-1,0,1,2):
            chip=replace(c,details=dict(c.details)|{'l2_bytes':r+w+delta})
            q=scenario([rows,cols],chip,base,mixed_bus_efficiency=.737)
            vals.append(q['latency_us'])
            self.assertLess(q['write_mixed_bytes'],1.)
        self.assertLess(max(vals)-min(vals),1.e-5)


if __name__=='__main__':unittest.main()
