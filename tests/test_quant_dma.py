from dataclasses import replace
from unittest import TestCase

from operator_model.chips import load_chip
from operator_model.operators.quant.resources import resource_quant
from operator_model.operators.quant.scale import peak_scale_writers


class TargetQuantDma(TestCase):
    def test_scale_owners(self):
        for rows,cols,r,c in [(8,384,1,256),(65,256,2,256),(36,512,1,512),(288,7168,36,1024),(128,5120,3,5120)]:
            row=cols//32
            expected=max(len({(i//row//r,(i%row)//(c//32)) for i in range(b,min(rows*row,b+32))})
                         for b in range(0,rows*row,32))
            self.assertEqual(peak_scale_writers(rows,cols,r,c),expected)

    def test_unknown_target_uses_requests_and_not_sku_name(self):
        chip=replace(load_chip(),name='future',vector_cores=80,hbm_bandwidth_gbs=3000)
        out=resource_quant((288,7168),chip)
        self.assertEqual(out,resource_quant((288,7168),replace(chip,name='unnamed')))
        tile=out[-1]['first_tile_cycles']
        self.assertEqual(tile['read_latency'],max(tile['local_read_cycles'],tile['shared_read_floor_cycles']))
        self.assertGreaterEqual(tile['write'],tile['scale_issue_gap_cycles']+tile['scale_write_cycles'])
        self.assertGreaterEqual(tile['write'],tile['data_write_cycles'])
        self.assertLess(tile['write'],tile['scale_write_cycles']+tile['data_write_cycles'])

    def test_shared_backpressure(self):
        chip=load_chip()
        fast=resource_quant((1024,4096),chip)
        slow=resource_quant((1024,4096),replace(chip,hbm_bandwidth_gbs=20))
        self.assertGreater(slow[0],fast[0])
        self.assertGreater(slow[-1]['chip_io_floor_cycles'],fast[-1]['chip_io_floor_cycles'])

    def test_local_read_issue_and_shared_completion_are_independent(self):
        chip=load_chip('Ascend950PR_957b')
        shape=(448,6592)
        base=resource_quant(shape,chip)
        slow=resource_quant(shape,replace(chip,hbm_bandwidth_gbs=180))
        self.assertEqual(base[-1]['first_tile_cycles']['read_ii'],slow[-1]['first_tile_cycles']['read_ii'])
        self.assertGreater(slow[-1]['read_plus_tail_floor_cycles'],base[-1]['read_plus_tail_floor_cycles'])
        wt=resource_quant(shape,replace(chip,details={**chip.details,'quant_output_retained_fraction':0} ))
        self.assertEqual(wt[-1]['assumed_offchip_write_bytes'],base[-1]['logical_write_bytes'])
        self.assertGreater(wt[0],base[0])
        saturated=resource_quant((262144,4096),chip)[-1]
        self.assertGreater(saturated['assumed_offchip_write_bytes'],0)
        self.assertLessEqual(saturated['output_retained_bytes'],chip.details['l2_bytes'])
