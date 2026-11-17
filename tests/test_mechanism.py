import copy
import unittest

from operator_model.extensions import plan
from operator_model.mechanism import build_graph, predict_graph, predict_plan, validate_profile, RESOURCE_UNITS, expand_instructions, extract_opcode_evidence, plan_from_case, predict_case


def profile(units=4, clock=1.):
    # Explicit synthetic rates for scheduler tests, not an Ascend profile.
    return {'name':'synthetic_test_fixture', 'resources':{
        name:dict(pool='memory' if name=='memory' else 'cube' if name.startswith('cube') else 'vector',
                  units=units,clock_ghz=clock,work_per_cycle=1.,latency_cycles=10.,issue_cycles=1.,
                  work_unit=unit,evidence='microbenchmark',source='synthetic unit-test fixture')
        for name,unit in RESOURCE_UNITS.items()}}


def graph(stages):
    return {'op':'synthetic', 'qualified_mapping':True,'stages':stages,'missing_evidence':[]}


def node(sid, resource='vector_alu', work=100, tasks=1, steps=1, serial=False, deps=()):
    return dict(id=sid,resource=resource,work=work,parallel_tasks=tasks,
                serial_steps=steps,state_serial=serial,dependencies=list(deps))


class TestMechanism(unittest.TestCase):
    def test_state_chain_cannot_be_parallelized_across_tokens(self):
        p=profile(units=64)
        pipelined=predict_graph(graph([node('a',work=8,tasks=1,steps=8)]),p)
        dependent=predict_graph(graph([node('a',work=8,tasks=1,steps=8,serial=True)]),p)
        self.assertAlmostEqual(pipelined['predicted_us'],.017)
        self.assertAlmostEqual(dependent['predicted_us'],.08)
        self.assertEqual(dependent['stages'][0]['active_units'],1)

    def test_dependency_and_shared_pool_contention(self):
        p=profile()
        independent=graph([node('cube','cube_bf16'),node('vec')])
        chained=graph([node('cube','cube_bf16'),node('vec',deps=['cube'])])
        same_pool=graph([node('a'),node('b','vector_exp')])
        self.assertAlmostEqual(predict_graph(independent,p)['predicted_us'],.1)
        self.assertAlmostEqual(predict_graph(chained,p)['predicted_us'],.2)
        self.assertAlmostEqual(predict_graph(same_pool,p)['predicted_us'],.2)

    def test_clock_and_core_count_are_separate_parameters(self):
        g=graph([node('a',work=800,tasks=8)])
        self.assertAlmostEqual(predict_graph(g,profile(4,1))['predicted_us'],.2)
        self.assertAlmostEqual(predict_graph(g,profile(8,1))['predicted_us'],.1)
        self.assertAlmostEqual(predict_graph(g,profile(4,2))['predicted_us'],.1)
        narrow=graph([node('a',work=800,tasks=1)])
        self.assertEqual(predict_graph(narrow,profile(4))['predicted_us'],predict_graph(narrow,profile(8))['predicted_us'])

    def test_missing_resource_is_not_free_and_hypothesis_not_qualified(self):
        p=profile();del p['resources']['vector_exp']
        result=predict_graph(graph([node('a','vector_exp'),node('b',deps=['a'])]),p)
        self.assertIsNone(result['candidate_us'])
        self.assertIsNone(result['predicted_us'])
        p=profile();p['resources']['vector_alu']['evidence']='hypothesis'
        result=predict_graph(graph([node('a')]),p)
        self.assertIsNotNone(result['candidate_us'])
        self.assertIsNone(result['predicted_us'])

    def test_reject_untraceable_coefficients_and_units(self):
        p=profile();p['per_operator_intercepts']={'rmsnorm':3}
        with self.assertRaises(ValueError):validate_profile(p)
        p=profile();p['resources']['vector_alu']['work_unit']='flops'
        with self.assertRaises(ValueError):validate_profile(p)
        with self.assertRaises(ValueError):predict_graph(graph([node('a',deps=['future'])]),profile())

    def test_existing_plans_get_qualified_limits(self):
        for op,shape,attrs in [
            ('rmsnorm',(4,128),{}),('addmm',(4,64,128),{}),
            ('lightning_indexer',(1,1,512,8,128),{}),('mhc_sinkhorn',(2,4),{}),
            ('mhc_post',(2,4,128),{}),('mhc_pre',(1,4,128),{}),
            ('mhc_pre_sinkhorn',(1,4,4096),{}),
            ('kda_recurrent',(1,2,1,1,128,128),{}),('kda_chunk',(1,32,1,1,16,16),{}),
            ('moe_ffn',(4,64,64,2),{'expert_counts':[1,3]})]:
            with self.subTest(op=op):
                structure=plan(op,shape,**attrs);before=copy.deepcopy(structure)
                result=predict_plan(structure,profile())
                self.assertIsNone(result['predicted_us'])
                self.assertGreater(result['candidate_us'],0)
                self.assertTrue(result['missing_evidence'])
                self.assertEqual(structure,before)

    def test_recurrence_tasks_and_skew_preserved(self):
        g=build_graph(plan('kda_recurrent',(2,8,1,4,128,128)))
        stage=next(s for s in g['stages'] if s['id']=='state_recurrence')
        self.assertEqual((stage['parallel_tasks'],stage['serial_steps'],stage['state_serial']),(8,8,True))
        g=build_graph(plan('moe_ffn',(8,64,64,4),expert_counts=[1,0,7,0]))
        first=[s for s in g['stages'] if s['id'].endswith('gmm1')]
        self.assertEqual([s['parallel_tasks'] for s in first],[1,7])
        self.assertEqual(sum(s['work'] for s in first),4*8*64*64)

    def test_real_manifest_adapters_preserve_flags_and_partial_graph(self):
        import json
        from pathlib import Path
        hardware=json.loads(Path('configs/hardware-950dt-v7.json').read_text())
        for path in Path('configs').glob('extension-*-v7.json'):
            for case in json.loads(path.read_text())['cases']:
                with self.subTest(case=case['id']):
                    result=predict_case(case['op'],case['shape'],case['attrs'],hardware)
                    self.assertNotEqual(result['structural_plan'].get('status'),'unsupported')
                    self.assertEqual(result['structural_plan']['call_attrs'],case['attrs'])
                    self.assertTrue(result['graph']['stages'])
                    self.assertIsNone(result['predicted_us'])
                    self.assertTrue(result['missing_evidence'])
        gmm2=plan_from_case('gmm2',(4,64,128,2),{'expert_counts':[1,3]})
        self.assertNotIn('gmm1_useful_flops',gmm2['work'])
        self.assertEqual(gmm2['work']['gmm2_useful_flops'],2*4*128*64)

    def test_unknown_semantic_flags_are_rejected_not_dropped(self):
        p=plan_from_case('rmsnorm',(1,128),{'mystery_fusion':True})
        self.assertEqual(p['status'],'unsupported')
        self.assertEqual(p['call_attrs'],{'mystery_fusion':True})
        p=plan_from_case('kda_recurrent',(1,1,1,1,128,128),{'use_gate_in_kernel':True})
        self.assertTrue(any('raw gate' in x for x in p['adapter_missing_evidence']))

    def test_source_instruction_expansion_preserves_dependencies_and_padding(self):
        g=graph([node('a',work=65,tasks=1),node('b',work=64,tasks=1,deps=['a'])])
        expanded=expand_instructions(g,{'vector_alu':{
            'source':'synthetic source template', 'evidence':'source',
            'instructions_per_work_unit':{'isa:RV_ADD':1/64}}})
        self.assertEqual([s['work'] for s in expanded['stages']],[2,1])
        self.assertEqual(expanded['stages'][1]['dependencies'],['a/isa0'])
        p=profile();p['resources']['isa:RV_ADD']=dict(p['resources']['vector_alu'],work_unit='instructions')
        self.assertGreater(predict_graph(expanded,p)['predicted_us'],0)

    def test_opcode_evidence_is_not_peak_throughput(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.db'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE KernelInfo(KernelInfoAttr TEXT,KernelInfoVal TEXT)')
                db.executemany('INSERT INTO KernelInfo VALUES (?,?)',[
                    ('helper_version','0.8.0'),('contract_version','7'),('frequency_ghz','1.65'),
                    ('name','synthetic'),('instrs_exec_num','2'),('exec_time_ticks','15')])
                db.execute('CREATE TABLE ExecutedInstructions(ExecInstrName TEXT,CoreTypeId INTEGER,ExecInstrTickStart INTEGER,ExecInstrTickEnd INTEGER,ExecInstrIsDurationExcluded INTEGER)')
                db.executemany('INSERT INTO ExecutedInstructions VALUES (?,?,?,?,?)',[
                    ('RV_ADD',2,0,10,0),('RV_ADD',2,5,15,0)])
            evidence=extract_opcode_evidence([path])['opcode_observations'][0]
            self.assertEqual(evidence['instruction_records'],2)
            self.assertEqual(evidence['duration_ticks_min'],10)
            self.assertIsNone(evidence['issue_cycles'])
            self.assertIsNone(evidence['work_per_cycle'])
            with self.assertRaises(ValueError):extract_opcode_evidence([path,path])

    def test_no_calibration_or_trace_fields_are_consumed(self):
        p=plan('rmsnorm',(4,128));a=predict_plan(p,profile())
        p.update(predicted_us=123456,reference_us=999999,coefficients=[999])
        self.assertEqual(a,predict_plan(p,profile()))


if __name__=='__main__':unittest.main()
