"""Holdout admission checks; failed/partial captures never acquire a CA error."""
from datetime import datetime
import math


def evaluate_record(frozen_case, record, frozen, frozen_sha):
    case=record['case']
    for key in ('id','op','shape','split','timeout_seconds'):
        if case[key]!=frozen_case[key]:raise ValueError('Frozen case mismatch: '+key)
    if case['split']!='holdout':raise ValueError('Expected independent holdout')
    if record['identity']['model_sha256']!=frozen['model_sha256']:raise ValueError('Model identity mismatch')
    if record['identity']['frozen_predictions_sha256']!=frozen_sha:raise ValueError('Freeze identity mismatch')
    if record['identity']['host_binary_sha256']!=frozen['artifact_sha256']['build/smoke/three_ops']:raise ValueError('Caller changed')
    if record['identity']['sku_adapter_sha256']!=frozen['artifact_sha256']['build/capture/libsim_soc_adapter.so']:raise ValueError('SKU adapter changed')
    if record['identity']['soc_version']!='Ascend950DT_9582' or record['identity']['cann_version']!='9.2.1':raise ValueError('Environment changed')
    if datetime.fromisoformat(record['started_utc'])<=datetime.fromisoformat(frozen['frozen_utc']):raise ValueError('Not post-freeze')
    if record['status']!='ok' or record['returncode']!=0 or record['timed_out']:return None
    m=record['measurement'];ref=m['reference_us'];new=frozen_case['prediction'];old=frozen_case['v04_prediction']
    if not math.isfinite(ref) or ref<=0:raise ValueError('Invalid CA reference')
    if m['reference_scope']!='cross-core instruction makespan':raise ValueError('Reference scope changed')
    if not m['correctness'] or not all('bad=0' in x for x in m['correctness']):raise ValueError('Missing correctness proof')
    return {'id':case['id'],'op':case['op'],'shape':case['shape'],'reference_us':ref,
        'v05_us':new['predicted_us'],'v04_us':old['predicted_us'],'constant_us':old['constant_baseline_us'],
        'v05_ape':abs(new['predicted_us']-ref)/ref,'v04_ape':abs(old['predicted_us']-ref)/ref,
        'constant_ape':abs(old['constant_baseline_us']-ref)/ref,'signed_error_us':new['predicted_us']-ref,
        'observed_cores':[m['active_aic'],m['active_aiv']],
        'predicted_cores':[new['plan']['active_aic'],new['plan']['active_aiv']],
        'predicted_mmad':new['estimated_max_core_selected_opcode_count'],'observed_mmad':m['max_core_mmad'],
        'kernel':m['kernel'],'db_sha256':m['db_sha256'],'correctness':m['correctness'],
        'phase_schedule':new['phase_schedule'],
        'ca_wall_seconds':record['wall_seconds'],'ca_cpu_seconds':record['cpu_seconds'],'raw_bytes':record['raw_bytes'],
        'extraction_seconds':record['extraction_seconds']}
