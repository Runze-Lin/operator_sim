"""Admission and leakage regressions for local v7 models; no CA or device needed."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from operator_model import model_v7, npusim_v7
from operator_model.ca import sha256

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIT = load_script('fit-v7')
COLLECT = load_script('collect-v7')


def route(attrs=None, domain=None):
    return {'attrs': attrs or {}, 'domain': domain or [[1, 128], [128, 256]],
            'parameters': {'startup_us': 2., 'service_us_per_work': .5,
                           'constant_us': 3., 'work_identifiable': True},
            'scope': 'cross-core instruction makespan',
            'calibration_work_min': 1, 'calibration_work_max': 4}


def predictor(routes):
    with patch.object(model_v7, 'BasePredictor'):
        return model_v7.Predictor({'schema_version': 7, 'base_v6': {}, 'routes': routes})


def make_db(path, updates=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {'helper_version': '0.8.0', 'contract_version': '7', 'frequency_ghz': '1.65',
            'instrs_exec_num': '3', 'exec_time_ticks': '30', 'exec_time_ns': str(30 / 1.65),
            'ai_core_active': '2', 'name': 'test_kernel', 'block_dim': '1'}
    info.update(updates or {})
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE KernelInfo (KernelInfoAttr TEXT, KernelInfoVal TEXT)')
        db.executemany('INSERT INTO KernelInfo VALUES (?,?)', info.items())
        db.execute('''CREATE TABLE ExecutedInstructions (
          ExecInstrCoreId INTEGER, CoreTypeId INTEGER, ExecInstrName TEXT,
          ExecInstrTickStart INTEGER, ExecInstrTickEnd INTEGER,
          ExecInstrIsDurationExcluded INTEGER, ExtendParams TEXT)''')
        db.executemany('INSERT INTO ExecutedInstructions VALUES (?,?,?,?,?,?,?)', [
            (0, 1, 'MMAD', 0, 20, 0, '{"queue_type":"CUBE"}'),
            (1, 2, 'RV_MUL', 5, 25, 0, '{"queue_type":"VECTOR"}'),
            (1, 2, 'RV_ADD', 15, 30, 0, '{"queue_type":"VECTOR"}')])


def valid_log(groups=1):
    return ('runtime_soc=Ascend950DT_9582\n' + ''.join(
        f'correctness name=out{i} count=16 bad=0 max_abs_error=0\n' for i in range(groups))
        + 'SMOKE_PASS\nSimulation SUCCESS\n')


class NonnegativeFitTests(unittest.TestCase):
    def test_recovers_identifiable_startup_and_slope(self):
        fit = model_v7.fit_nonnegative([1., 2., 4.], [3., 5., 9.])
        self.assertAlmostEqual(fit['startup_us'], 1.)
        self.assertAlmostEqual(fit['service_us_per_work'], 2.)
        self.assertTrue(fit['work_identifiable'])

    def test_identical_per_core_work_falls_back_to_constant(self):
        fit = model_v7.fit_nonnegative([40., 40.], [3., 5.])
        self.assertFalse(fit['work_identifiable'])
        self.assertEqual(fit['service_us_per_work'], 0.)
        self.assertEqual(fit['startup_us'], 4.)

    def test_decreasing_labels_cannot_create_negative_service(self):
        fit = model_v7.fit_nonnegative([1., 2.], [5., 3.])
        self.assertEqual(fit['startup_us'], 4.)
        self.assertEqual(fit['service_us_per_work'], 0.)

    def test_negative_unconstrained_intercept_uses_boundary_fit(self):
        fit = model_v7.fit_nonnegative([1., 2.], [1., 4.])
        self.assertEqual(fit['startup_us'], 0.)
        self.assertAlmostEqual(fit['service_us_per_work'], 1.8)

    def test_invalid_or_nonfinite_observations_rejected(self):
        cases = [([1], [2]), ([1, 2], [3]), ([1, 2], [-1, -2]),
                 ([1, float('nan')], [2, 3]), ([1, 2], [2, float('inf')]),
                 ([-1, 2], [2, 3])]
        for xs, ys in cases:
            with self.subTest(xs=xs, ys=ys), self.assertRaises(ValueError):
                model_v7.fit_nonnegative(xs, ys)


class LocalContractTests(unittest.TestCase):
    def test_fused_pre_sinkhorn_has_flattened_public_hres_output(self):
        from operator_model.extensions.indexer_mhc import plan
        fused = plan('mhc_pre_sinkhorn', (2, 4, 4096))
        separate = plan('mhc_pre', (2, 4, 4096))
        self.assertEqual(fused['output_shapes']['h_res'], [2, 16])
        self.assertEqual(separate['output_shapes']['h_res'], [2, 4, 4])
        self.assertEqual(math.prod(fused['output_shapes']['h_res']),
                         math.prod(separate['output_shapes']['h_res']))

    def test_unknown_route_does_not_invent_latency(self):
        out = predictor({}).predict(model_v7.Case('unknown', (1, 2)))
        self.assertIsNone(out['predicted_us'])
        self.assertEqual(out['status'], 'unqualified_route')

    def test_outside_shape_or_attribute_contract_returns_null(self):
        p = predictor({'rmsnorm': route()})
        for shape, attrs in [((129, 128), {}), ((1, 127), {}), ((1, 128, 1), {}),
                             ((1, 128), {'dtype': 'fp16'}), ((1, 128), {'return_intermediates': True})]:
            with self.subTest(shape=shape, attrs=attrs):
                out = p.predict(model_v7.Case('rmsnorm', shape), attrs)
                self.assertIsNone(out['predicted_us'])
                self.assertEqual(out['status'], 'outside_calibrated_contract')

    def test_per_core_post_work_does_not_scale_with_active_core_count(self):
        work = model_v7.work
        self.assertEqual(work('mhc_post', (1, 4, 128), {}), work('mhc_post', (32, 4, 128), {}))
        self.assertEqual(work('mhc_post', (65, 4, 128), {}), 2 * work('mhc_post', (1, 4, 128), {}))

    def test_addmm_bias_shape_normalization_preserves_real_contract(self):
        p = predictor({'addmm': route({'self_shape': ['N'], 'alpha': 1, 'beta': 1},
                                     [[1, 16], [64, 128], [64, 128]])})
        args = model_v7.Case('addmm', (2, 128, 64))
        self.assertIsNotNone(p.predict(args, {'self_shape': [128], 'alpha': 1, 'beta': 1})['predicted_us'])
        self.assertIsNone(p.predict(args, {'self_shape': [2, 128], 'alpha': 1, 'beta': 1})['predicted_us'])

    def test_moe_requires_valid_expert_histogram(self):
        for counts in (None, [], [4], [2, 1], [-1, 5], [2., 2.]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                model_v7.work('gmm2', (4, 128, 64, 2), {'expert_counts': counts})

    def test_moe_composition_preserves_components_and_excludes_communication(self):
        p = predictor({name: route({}, [[1, 8], [64, 128], [64, 128], [1, 2]])
                       for name in ('grouped_matmul_swiglu_quant_v2', 'gmm2')})
        out = p.predict(model_v7.Case('moe_ffn', (4, 128, 64, 2)), {'expert_counts': [1, 3]})
        self.assertEqual(out['predicted_us'], sum(c['predicted_us'] for c in out['components']))
        self.assertEqual(out['constant_us'], 6.)
        self.assertIn('excludes routing/communication', out['scope'])
        self.assertIn('launch gaps', out['scope'])

    def test_moe_partial_component_calibration_does_not_become_full_latency(self):
        p = predictor({'gmm2': route({}, [[1, 8], [64, 128], [64, 128], [1, 2]])})
        out = p.predict(model_v7.Case('moe_ffn', (4, 128, 64, 2)), {'expert_counts': [1, 3]})
        self.assertIsNone(out['predicted_us'])
        self.assertEqual(out['status'], 'outside_calibrated_contract')


class DatabaseAdmissionTests(unittest.TestCase):
    def test_complete_db_has_cross_core_makespan_not_sum_of_core_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'valid.db'; make_db(path)
            out = npusim_v7.extract_db(path)
            self.assertAlmostEqual(out['reference_us'], 30 / 1650)
            self.assertEqual((out['active_aic'], out['active_aiv']), (1, 1))
            vec = next(c for c in out['cores'] if c['core_type_id'] == 2)
            self.assertAlmostEqual(vec['pipeline_union_us']['VECTOR'], 25 / 1650)
            self.assertEqual(out['max_core_mmad'], 1)
            self.assertEqual(out['max_core_rv_count'], 2)
            self.assertIsNone(out['kernel_binary_sha256'])

    def test_metadata_inconsistency_is_not_admitted(self):
        for updates in ({'helper_version': '0.7.0'}, {'contract_version': '6'},
                        {'frequency_ghz': '1.8'}, {'instrs_exec_num': '2'},
                        {'exec_time_ticks': '31'}, {'exec_time_ns': '99'}, {'ai_core_active': '1'}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'invalid.db'; make_db(path, updates)
                with self.assertRaises(ValueError):
                    npusim_v7.extract_db(path)

    def test_negative_duration_unknown_core_and_missing_queue_are_rejected(self):
        changes = ["UPDATE ExecutedInstructions SET ExecInstrTickEnd=-1 WHERE ExecInstrName='MMAD'",
                   'UPDATE ExecutedInstructions SET CoreTypeId=99 WHERE CoreTypeId=1',
                   "UPDATE ExecutedInstructions SET ExtendParams='{}'"]
        for sql in changes:
            with self.subTest(sql=sql), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'invalid.db'; make_db(path)
                with sqlite3.connect(path) as db:
                    db.execute(sql)
                with self.assertRaises(ValueError):
                    npusim_v7.extract_db(path)

    def test_no_record_database_is_not_a_zero_latency_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.db'; make_db(path)
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM ExecutedInstructions')
            with self.assertRaises(ValueError):
                npusim_v7.extract_db(path)

    def test_two_kernels_sum_is_explicitly_not_api_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('0000_first.db', '0001_second.db'):
                make_db(root / 'npusim_test' / 'record' / name)
            log = root / 'run.log'; log.write_text(valid_log(2))
            out = npusim_v7.extract_run(root, log, 2)
            self.assertAlmostEqual(out['reference_us'], 60 / 1650)
            self.assertEqual(out['kernel_count'], 2)
            self.assertIn('inter-launch gaps excluded', out['reference_scope'])
            self.assertIn('no cross-DB global-clock claim', out['scope_limit'])

    def test_stage_binding_uses_launch_id_not_database_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_db(root / 'npusim_test/record/0000_second.db', {'launch_id': '8', 'name': 'second_kernel'})
            make_db(root / 'npusim_test/record/0001_first.db', {'launch_id': '3', 'name': 'first_kernel'})
            log = root / 'run.log'
            markers = ('stage=grouped_matmul_swiglu_quant_v2 api=aclnnGroupedMatmulSwigluQuantV2\n'
                       'phase → task 3 done\nstage_complete=grouped_matmul_swiglu_quant_v2\n'
                       'stage=gmm2 api=aclnnQuantMatmulV5\n'
                       'phase → task 8 done\nstage_complete=gmm2\n')
            log.write_text(valid_log() + markers)
            out = npusim_v7.extract_run(root, log, 1)
            self.assertEqual([s['kernel'] for s in out['stages']], ['first_kernel', 'second_kernel'])
            self.assertEqual([s['launch_id'] for s in out['stages']], [3, 8])
            self.assertEqual(out['stages'][0]['db_sha256'], out['kernels'][1]['db_sha256'])

    def test_missing_duplicate_or_ambiguous_stage_kernels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_db(root / 'npusim_test/record/0000.db', {'launch_id': '3'})
            log = root / 'run.log'
            cases = [
                'stage=gmm2 api=API\nphase → task 3 done\n',
                'stage=gmm2 api=API\nphase → task 3 done\nphase → task 3 done\nstage_complete=gmm2\n',
                'stage=gmm2 api=API\nphase → task 8 done\nstage_complete=gmm2\n',
                ('stage=gmm2 api=API\nphase → task 3 done\nstage_complete=gmm2\n') * 2,
            ]
            for marker in cases:
                with self.subTest(marker=marker):
                    log.write_text(valid_log() + marker)
                    with self.assertRaises(ValueError):
                        npusim_v7.extract_run(root, log, 1)

    def test_raw_log_stage_mapping_allows_duplicate_task_done_and_reused_task_id_on_new_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'npusim_test'
            make_db(archive / 'record/0000_z.db', {'launch_id': '1', 'name': 'second'})
            make_db(archive / 'record/0001_a.db', {'launch_id': '0', 'name': 'first'})
            raw = archive / 'npusim.log'
            first = '[info] [0000012523] [TASK_DONE] RTSQ_2 ACSQ_0 stream_id:0, task_id:0 TASK_DONE\n'
            second = '[info] [0000020276] [TASK_DONE] RTSQ_2 ACSQ_0 stream_id:1, task_id:0 TASK_DONE\n'
            raw.write_text('stage=grouped_matmul_swiglu_quant_v2 api=First\n' + first * 2
                           + 'stage_complete=grouped_matmul_swiglu_quant_v2\n'
                           + 'stage=gmm2 api=Second\n' + second * 2 + 'stage_complete=gmm2\n')
            log = root / 'run.log'
            # CLI phase IDs are not reliable launch identities; raw log takes precedence.
            log.write_text(valid_log() + 'stage=bad_cli api=Wrong\nphase → task 999 done\n')
            out = npusim_v7.extract_run(root, log, 1)
            self.assertEqual([s['kernel'] for s in out['stages']], ['first', 'second'])
            self.assertEqual([s['launch_id'] for s in out['stages']], [0, 1])
            self.assertEqual(out['stages'][0]['runtime_tasks'], [('0', '0')])
            self.assertEqual(out['stages'][1]['runtime_tasks'], [('1', '0')])
            self.assertTrue(all('synchronized API stage' in s['attribution'] for s in out['stages']))

    def test_raw_stage_with_two_distinct_runtime_tasks_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'npusim_test'
            make_db(archive / 'record/0000.db', {'launch_id': '0'})
            raw = archive / 'npusim.log'
            raw.write_text('stage=gmm2 api=API\n'
                           '[TASK_DONE] stream_id:0, task_id:0 TASK_DONE\n'
                           '[TASK_DONE] stream_id:0, task_id:1 TASK_DONE\n'
                           'stage_complete=gmm2\n')
            log = root / 'run.log'; log.write_text(valid_log())
            with self.assertRaisesRegex(ValueError, 'exactly one attributable runtime task'):
                npusim_v7.extract_run(root, log, 1)

    def test_raw_stage_mapping_does_not_silently_drop_unattributed_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'npusim_test'
            make_db(archive / 'record/0000.db', {'launch_id': '0'})
            make_db(archive / 'record/0001.db', {'launch_id': '1'})
            (archive / 'npusim.log').write_text('stage=gmm2 api=API\n'
                '[TASK_DONE] stream_id:0, task_id:0 TASK_DONE\nstage_complete=gmm2\n')
            log = root / 'run.log'; log.write_text(valid_log())
            with self.assertRaisesRegex(ValueError, 'Unattributed or duplicated stage kernels'):
                npusim_v7.extract_run(root, log, 1)

    def test_raw_stage_without_runtime_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'npusim_test'
            make_db(archive / 'record/0000.db', {'launch_id': '0'})
            (archive / 'npusim.log').write_text('stage=gmm2 api=API\nstage_complete=gmm2\n')
            log = root / 'run.log'; log.write_text(valid_log())
            with self.assertRaisesRegex(ValueError, 'exactly one attributable runtime task'):
                npusim_v7.extract_run(root, log, 1)

    def test_multiple_raw_runtime_logs_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('npusim_a', 'npusim_b'):
                make_db(root / directory / 'record/0000.db', {'launch_id': '0'})
                (root / directory / 'npusim.log').write_text('raw log')
            log = root / 'run.log'; log.write_text(valid_log())
            with self.assertRaisesRegex(ValueError, 'Ambiguous runtime log'):
                npusim_v7.extract_run(root, log, 1)

    def test_partial_failed_wrong_soc_and_empty_correctness_logs_rejected(self):
        log_cases = [valid_log().replace('Simulation SUCCESS', ''),
                     valid_log() + 'SMOKE_FAIL bad output\n',
                     valid_log().replace('Ascend950DT_9582', 'Ascend910B'),
                     valid_log().replace('bad=0', 'bad=1'),
                     valid_log().replace('count=16', 'count=0'),
                     valid_log().replace('SMOKE_PASS', '')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_db(root / 'npusim_test' / 'record' / '0000.db')
            log = root / 'run.log'
            for text in log_cases:
                with self.subTest(text=text):
                    log.write_text(text)
                    with self.assertRaises(ValueError):
                        npusim_v7.extract_run(root, log, 1)
            log.write_text(valid_log())
            with self.assertRaises(ValueError):
                npusim_v7.extract_run(root, log, 2)


class FreezeAndAttributionTests(unittest.TestCase):
    def test_fit_refuses_any_already_observed_holdout_even_if_failed(self):
        for status in ('ok', 'failed'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'out'
                with patch.object(FIT, 'OUT', out), patch.object(FIT, 'records', return_value=[
                    {'case': {'split': 'holdout'}, 'status': status}]):
                    with self.assertRaisesRegex(ValueError, 'Holdout already observed'):
                        FIT.main()
                self.assertFalse((out / 'model.json').exists())

    def test_existing_freeze_cannot_be_overwritten_by_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / 'frozen-predictions.json').write_text('frozen-marker')
            with patch.object(FIT, 'OUT', out), patch.object(FIT, 'records') as read_records:
                with self.assertRaisesRegex(ValueError, 'Already frozen'):
                    FIT.main()
                read_records.assert_not_called()
            self.assertEqual((out / 'frozen-predictions.json').read_text(), 'frozen-marker')

    def test_failed_calibration_never_becomes_observation(self):
        self.assertEqual(FIT.observations([{'status': 'failed'}]), [])

    def test_moe_two_arbitrary_kernels_cannot_be_attributed_by_position(self):
        record = {'status': 'ok', 'record_path': 'mock.json',
                  'case': {'id': 'moe_case', 'op': 'moe_ffn', 'split': 'calibration',
                           'components': ['grouped_matmul_swiglu_quant_v2', 'gmm2']},
                  'measurement': {'reference_us': 3., 'reference_scope': 'serial sum',
                                  'kernels': [{'kernel': 'ViewCopy', 'reference_us': 1., 'reference_scope': 'kernel'},
                                              {'kernel': 'Unknown', 'reference_us': 2., 'reference_scope': 'kernel'}]}}
        with self.assertRaises(ValueError):
            FIT.observations([record])

    def test_moe_attribution_tracks_stage_identity_and_preserves_holdout_role(self):
        components = ['grouped_matmul_swiglu_quant_v2', 'gmm2']
        first = {'kernel': 'first', 'reference_us': 1., 'reference_scope': 'kernel',
                 'db_sha256': 'first-sha', 'helper_info': {'launch_id': '3'}}
        second = {'kernel': 'second', 'reference_us': 2., 'reference_scope': 'kernel',
                  'db_sha256': 'second-sha', 'helper_info': {'launch_id': '8'}}
        stages = [{'op': op, 'db_sha256': k['db_sha256'], 'launch_id': int(k['helper_info']['launch_id'])}
                  for op, k in zip(components, [first, second])]
        record = {'status': 'ok', 'record_path': 'mock.json',
                  'case': {'id': 'h', 'op': 'moe_ffn', 'split': 'holdout', 'components': components},
                  'measurement': {'reference_us': 3., 'reference_scope': 'serial sum',
                                  'kernels': [second, first], 'stages': stages}}
        out = FIT.observations([record])
        self.assertEqual([row['reference_us'] for row in out], [3., 1., 2.])
        self.assertTrue(all(row['case']['split'] == 'holdout' for row in out))
        self.assertTrue(all(row['parent_id'] == 'h' for row in out))
        record['measurement']['stages'][0]['db_sha256'] = 'wrong-binary'
        with self.assertRaisesRegex(ValueError, 'Ambiguous stage identity'):
            FIT.observations([record])

    def check_holdout_preflight(self, mutate_case=False, mutate_binary=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'runs').mkdir()
            for name in ('bin/test', 'src/test.cpp', 'build/capture/libsim_soc_adapter.so', 'operator_model/npusim_v7.py'):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original')
            case = {'id': 'h', 'op': 'rmsnorm', 'shape': [1, 128], 'attrs': {},
                    'split': 'holdout', 'timeout_seconds': 1, 'arguments': ['rmsnorm', '1', '128'],
                    'correctness_groups': 1}
            spec = {'binary': 'bin/test', 'source': 'src/test.cpp', 'cases': [case]}
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps(spec))
            frozen_case = copy.deepcopy(case)
            if mutate_case:
                frozen_case['shape'] = [1, 256]
            frozen = {'predictions': [{'id': 'h', 'case': frozen_case}], 'model_sha256': 'test-model',
                      'artifact_sha256': {'bin/test': sha256(root / 'bin/test')}}
            fp = root / 'results/v0.7/frozen-predictions.json'; fp.parent.mkdir(parents=True); fp.write_text(json.dumps(frozen))
            if mutate_binary:
                (root / 'bin/test').write_text('changed-after-freeze')
            with patch.object(COLLECT, 'ROOT', root), patch('sys.argv', ['collect-v7.py', '--manifest', 'manifest.json', '--case', 'h']), \
                    patch.object(COLLECT.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'Case changed|Frozen artifact changed'):
                    COLLECT.main()
                launch.assert_not_called()
            self.assertFalse((root / 'runs/model-v7/h').exists())

    def test_changed_holdout_case_is_rejected_before_launch(self):
        self.check_holdout_preflight(mutate_case=True)

    def test_changed_frozen_binary_is_rejected_before_launch(self):
        self.check_holdout_preflight(mutate_binary=True)


if __name__ == '__main__':
    unittest.main()
