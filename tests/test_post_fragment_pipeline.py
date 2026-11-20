import unittest
from operator_model import OperatorStream
from operator_model.post_fragment_pipeline import _advance_rows, estimate_post_fragments


class PostFragmentTest(unittest.TestCase):
    def test_jump_matches_row_by_row(self):
        for c0,c1 in [((19,30),(27,37)),((30,95),(18,33)),((20,9),(45,140))]:
            a,b = [dict(ii=i,slot_path=s) for i,s in (c0,c1)]
            pair=(183,151)
            for n in range(37):
                self.assertEqual(_advance_rows((183,151),a,b,n),pair)
                w0=max(pair[0]+a['ii'],pair[1]+a['slot_path'])
                w1=max(w0+b['ii'],pair[0]+b['slot_path'])
                pair=w1,w0

    def test_tail_and_release(self):
        p=estimate_post_fragments([67,4,4912],chip='Ascend950DT_9582',resident_aiv=56,previous_shape=[112,4,768],inspect_rows=[0,1])
        self.assertEqual([r['columns']for r in p['timeline']],[2464,2448]*2)
        for r in p['timeline']:
            self.assertLessEqual(r['x_read_us'][1],r['write_issue_us'])
        self.assertLessEqual(p['timeline'][-1]['final_write_ack_us'],p['us'])
        self.assertAlmostEqual(sum(p['stages_us'].values()),p['us'])

    def test_eligibility_and_work_cost(self):
        args=dict(chip='Ascend950DT_9582',resident_aiv=64,previous_shape=[121,4,1216])
        for shape in ([3,4,5120],[65,4,1536]):
            with self.assertRaises(ValueError):estimate_post_fragments(shape,**args)
        shape=[65,4,5120]
        for bad in (dict(previous_shape=shape),dict(resident_aiv=0), dict(previous_shape=[1,4,7168]),dict(inspect_rows=[2])):
            with self.assertRaises(ValueError):estimate_post_fragments(shape,**{**args,**bad})
        p=estimate_post_fragments([262144,4,5120],**args,inspect_rows=[4095])
        self.assertEqual(len(p['timeline']),2)
        self.assertGreater(p['us'],estimate_post_fragments(shape,**args)['us'])

    def test_default_ordered_stream_reuses_successful_dispatch_only(self):
        shape=[65,4,5120];prev=[121,4,1216]
        def model(x):return x.projection['timing_diagnostics']['model']
        legacy=OperatorStream(track_post_history=True,experimental_post_fragments=False)
        new=OperatorStream()
        for s in (legacy,new):s.advance('mhc_post',prev)
        prior=new.preview('mhc_post',shape)
        self.assertEqual(model(prior),'post_two_fragment_slot_reuse')
        self.assertEqual(model(legacy.preview('mhc_post',shape)),'post_target_critical_aiv')
        new.preview('matmul',[16,64,256]);new.advance('mhc_post',shape,dtype='fp16')
        self.assertEqual(new.preview('mhc_post',shape),prior)
        self.assertNotEqual(model(new.preview('mhc_post',shape,{'resident_aiv':0})),model(prior))
        new.advance('mhc_post',shape)
        self.assertEqual(model(new.preview('mhc_post',shape)),'post_token_immediate_repeat')
        new.forget_history()
        self.assertNotEqual(model(new.preview('mhc_post',shape)),model(prior))
        with self.assertRaises(TypeError):OperatorStream(experimental_post_fragments=1)
