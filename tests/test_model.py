import json
import unittest

from operator_model.ca import union_length
from operator_model.model import Case, Predictor, fit, fit_nonnegative_line, plan, quant_partition, schedule


def synthetic_training():
    """Synthetic tests verify contracts only; accuracy comes from CA holdouts."""
    result=[]
    for op,shapes in {'quant':[(4,256),(17,512),(64,1024)],
                      'matmul':[(32,64,64),(1,128,128),(128,256,512)],
                      'fa':[(16,64,64),(1,128,64),(64,256,64)]}.items():
        for i,shape in enumerate(shapes):
            p=plan(Case(op,shape))
            result.append({'status':'ok','case':{'id':f'{op}{i}','op':op,'shape':shape,'split':'calibration'},
                'measurement':{'reference_us':3+i,'kernel':op,'active_aic':p['active_aic'],'active_aiv':p['active_aiv'],
                    'cores':[{'kind':'AIC' if op=='matmul' else 'AIV', 'dynamic_opcodes':{'MMAD':1,'RV_ADD':3},
                              'busy_union_us_by_pipe':{'MTE2':.4,'VECTOR_FAMILY':.1,'CUBE':.05,'FIXP':.2,'MTE3':.3}}]}})
    return result


class ModelTests(unittest.TestCase):
    def test_union_overlap_and_zero_duration(self):
        self.assertEqual(union_length([(0,3),(1,2),(2,4),(6,7),(7,7)]),5)
        self.assertEqual(union_length([]),0)
        with self.assertRaises(ValueError):union_length([(4,1)])

    def test_quant_coverage_and_capacity(self):
        for rows in (1,4,17,33,64,129):
            for cols in (64,256,384,512,768,1024):
                p=quant_partition(rows,cols)
                self.assertLessEqual(p['active_aiv'],64)
                self.assertGreaterEqual(p['active_aiv']*p['rows_per_core']*p['cols_per_core'],rows*cols)
        self.assertEqual(quant_partition(17,512)['active_aiv'],34)

    def test_mapping_tail_and_wave(self):
        p=plan(Case('matmul',(17,96,128)))
        self.assertEqual(p['active_aic'],4)
        self.assertGreaterEqual(p['active_aic']*p['work_per_core'],p['useful_work'])
        self.assertGreater(plan(Case('matmul',(1024,1024,64)))['waves'],1)

    def test_fa_dependencies_not_max_only(self):
        p={'waves':1,'iterations':2}
        costs={'read':1.,'cube':4.,'vector':3.,'fix':0.,'write':1.}
        self.assertEqual(schedule('fa',p,costs),14.)  # fill 8 + recurrence 5 + drain 1

    def test_fit_boundaries(self):
        self.assertEqual(fit_nonnegative_line([1,2,3],[4,3,2]),{'intercept_us':3.,'slope_us_per_unit':0.})
        self.assertEqual(fit_nonnegative_line([1,2,3],[3,5,7]),{'intercept_us':1.,'slope_us_per_unit':2.})

    def test_reject_leakage_and_missing_family(self):
        data=synthetic_training()
        data[0]['case']['split']='holdout'
        with self.assertRaises(ValueError):fit(data,'test')
        with self.assertRaises(ValueError):fit(synthetic_training()[:-1],'test')

    def test_domain(self):
        for op,shape in [('quant',(1,33)),('fa',(1,128,128)),('matmul',(0,64,64)),('quant',(1.,64))]:
            with self.assertRaises(ValueError):Case(op,shape)

    def test_cache_equivalence_immutability_and_bound(self):
        predictor=Predictor(fit(synthetic_training(),'test'),cache_size=2)
        case=Case('quant',(4,256))
        a=predictor.predict(case)
        self.assertEqual(a,predictor.predict(case,use_cache=False))
        self.assertIs(a,predictor.predict(case))
        with self.assertRaises(TypeError):a['plan']['active_aiv']=99
        json.dumps(a)
        predictor.predict(Case('quant',(8,256)))
        predictor.predict(Case('quant',(16,256)))
        self.assertEqual(len(predictor.cache),2)
        self.assertEqual(a,predictor.predict(case))

    def test_extrapolation_and_batch(self):
        predictor=Predictor(fit(synthetic_training(),'test'))
        case=Case('quant',(100,2048))
        self.assertEqual(predictor.predict(case)['coverage'],'extrapolation')
        self.assertFalse(predictor.predict(case)['kernel_selection_verified'])
        self.assertIsNone(predictor.predict(case)['ideal_compute_efficiency'])
        self.assertEqual(len(predictor.predict_many([case,case])),2)


if __name__=='__main__':unittest.main()
