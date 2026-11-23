"""Exercise archive qualification with compact receipts from the published M=9 case."""
import copy
import unittest

from calibration.__main__ import HERE, digest, read, seal
from calibration.mx_matmul.compare import compare
from calibration.verify import verify_mx_projection_rows


def reseal(document):
    return seal({k:v for k,v in document.items() if k!='sha256'})


def example_archive():
    # Rebind copies in memory to build a two-version protocol fixture.
    historical=read(HERE/'reference/mx-matmul.json')['independent']
    original=historical['freeze']
    cases=[c for c in original['cases'] if c['shape']==[9,128,256]]
    archive={'geometries':[{'id':'p01','shape':cases[0]['shape']}],
             'budget':{'points':4,'geometries_per_route':1,'observed_before_new_acquisition':0},
             'observations':[]}
    for version in ('1.2','1.3'):
        frozen=reseal(dict(original,model_version=version+'.0',package_sha256=digest(version.encode()),cases=cases))
        plans={}
        for c in cases:
            mixed=c['contract']=='matmul_mx_mixed'
            plan={'aic':8,'launched_aiv':16 if mixed else 0,'producer_aiv':8 if mixed else 0}
            plans[c['id']]={}
            for target,field in [('target','predicted_us'),('970dt','extrapolated_970dt_us')]:
                plans[c['id']][target]=dict(plan,status='extrapolated',us=c[field],stages_us={'window':c[field]})
        archive['v'+version]={'freeze':frozen, 'schedules':seal(dict(kind='mx-plan-freeze',schema=1,
            freeze_sha256=frozen['sha256'],created_utc=frozen['created_utc'],plans=plans))}
    ids={c['id'] for c in cases}
    for receipt in historical['receipts']:
        ident=receipt['case']['id']
        if ident in ids:
            archive['observations'].append({'receipt':dict(receipt,freeze_sha256=archive['v1.3']['freeze']['sha256']),
                'phases':historical['phase_counts'][ident]})
    return archive


class MxComparisonTests(unittest.TestCase):
    def setUp(self):
        self.archive=example_archive()

    def test_valid_pair_and_echoed_export(self):
        result, observed=compare(self.archive)
        self.assertEqual(len(result['pairs']),2)
        self.assertEqual(len(observed),4)
        rows=[dict(e['case'],**{k:e['measurement'][k] for k in
                   ('case_id','ca_us','provider','identity','kernel','freeze_sha256','receipt_sha256')},
                   data_role='independent',predicted_us=e['v1.3']['us'],extrapolated_970dt_us=e['v1.3_970dt']['us'])
              for e in observed.values()]
        verify_mx_projection_rows(observed,rows,set())
        for field,value in [('data_role','replay'),('extrapolated_970dt_us',0),('dtype','bf16')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'reference differs'):
                verify_mx_projection_rows(observed,[dict(rows[0],**{field:value}),*rows[1:]],set())

    def test_wrong_runtime_kernel_or_matrix_input(self):
        observations=self.archive['observations']
        native=next(o for o in observations if o['receipt']['case']['dtype']=='mxfp4')
        mixed=next(o for o in observations if o['receipt']['case']['dtype']=='fp8_e4m3fn')
        stage=native['receipt']['stages'][0]
        original=stage['kernel']
        stage['kernel']=mixed['receipt']['stages'][0]['kernel']
        with self.assertRaisesRegex(ValueError,'Runtime kernel'):
            compare(self.archive)
        stage['kernel']=original
        fingerprint=stage['kernel_binary_sha256']
        stage['kernel_binary_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Runtime kernel'):
            compare(self.archive)
        stage['kernel_binary_sha256']=fingerprint
        mixed['phases']['fragments'][0]['b']='fp4_e2m1'
        with self.assertRaisesRegex(ValueError,'MMAD fragments'):
            compare(self.archive)

    def test_wrong_version_binding_or_future_qualification(self):
        latest=self.archive['v1.3']
        original=latest['schedules']
        latest['schedules']=reseal(dict(original,freeze_sha256=self.archive['v1.2']['freeze']['sha256']))
        with self.assertRaisesRegex(ValueError,'freezes differ'):
            compare(self.archive)
        modified=copy.deepcopy(original)
        next(iter(modified['plans'].values()))['970dt']['status']='estimated'
        latest['schedules']=reseal(modified)
        with self.assertRaisesRegex(ValueError,'prices diverge'):
            compare(self.archive)


if __name__=='__main__':
    unittest.main()
