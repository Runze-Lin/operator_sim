from dataclasses import replace
import unittest
from operator_model import estimate, load_chip
from operator_model.rms_resource import resource_rms
from operator_model.rms_target_cost import resource_rms_target, RmsFirstDispatch


class RmsTargetCostTest(unittest.TestCase):
    def test_extra_prices_respect_state_grid_and_frequency(self):
        chip=load_chip('Ascend950PR_957b')
        shape=(8,6016)
        prices=RmsFirstDispatch(2580.5,2776.)
        self.assertEqual(resource_rms_target(shape,chip),resource_rms(shape,chip))
        us,_,a,diag=resource_rms_target(shape,chip,first_dispatch=prices)
        self.assertAlmostEqual(diag['stages_us']['SCALAR'],2580.5/1650)
        self.assertAlmostEqual(us,sum(diag['stages_us'].values()))
        slower=replace(chip,name='custom',frequency_ghz=chip.frequency_ghz/2)
        self.assertAlmostEqual(resource_rms_target(shape,slower,first_dispatch=prices)[3]['stages_us']['SCALAR'],2580.5/825)
        for resident in (1,a):
            warm=resource_rms_target(shape,chip,{'resident_aiv':resident},prices)
            self.assertEqual(warm[0],resource_rms(shape,chip,{'resident_aiv':resident})[0])

    def test_public_api_reads_two_independently_optional_chip_fields(self):
        chip=load_chip('Ascend950PR_957b')
        get=lambda c:estimate('rmsnorm',(9,5120),chip=c,timing_mode='resource')
        y=get(chip)
        self.assertEqual(get(replace(chip,name='anonymous')).latency_us,y.latency_us)
        self.assertEqual(y.projection['timing_diagnostics']['first_dispatch_cycles']['scalar'],2580.5)
        base=load_chip()
        for k in ['rms_first_scalar_cycles','rms_first_payload_cycles']:
            renamed=replace(base,details={**base.details,k:2500.})
            self.assertIsNotNone(get(renamed).latency_us)
        self.assertNotIn('first_dispatch_cycles',get(load_chip('Ascend970DT')).projection['timing_diagnostics'])

    def test_release_window_does_not_double_bill_scalar(self):
        chip=replace(load_chip(),hbm_bandwidth_gbs=1)
        shape=(128,4096)
        prices=RmsFirstDispatch(100,200.)
        us,_,_,diag=resource_rms_target(shape,chip,first_dispatch=prices)
        self.assertGreater(diag['stages_us']['shared_io'],0.)
        self.assertAlmostEqual(us,(200+diag['shared_io_floor_cycles'])/1650)
        with self.assertRaises(ValueError):RmsFirstDispatch(-1,200)
