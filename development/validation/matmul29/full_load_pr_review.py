"""Post-freeze residual audit for the first blinded large PR endpoint."""
from pathlib import Path
import json
from full_load_freeze import ROOT,OUT,read,digest
from full_load_trace import extract

def review():
    frozen=read(OUT/'freeze.json');case='262144-128-512-pr'
    observed=extract(case)
    f,=[r for r in frozen['rows'] if r['shape']==observed['shape'] and r['chip']==observed['chip']]
    assert f['role']=='in flight, blinded at freeze'
    p=f['candidate'];cold=p['work']['minimum_cold_hbm_bytes'];us=observed['ca_us'];gbs=1400.
    issue=frozen['utc']
    d=dict(role=f['role'],freeze_utc=issue,source=observed['drain_receipt'],receipt_sha256=observed['drain_sha256'],
           critical_core=observed['core'],shard_sha256=observed['core_sha256'],tiles=len(observed['tiles']),metrics=observed['metrics'],
           global_us=us,forecast_us=p['latency_us'],old_us=f['legacy_us'],hbm_gbs=gbs,cold_min_bytes=cold,
           optimistic_cold_floor_us=cold/(1000*gbs),effective_gbs_if_all_cold=cold/(us*1000),
           min_non_hbm_bytes_at_configured_peak=cold-us*1000*gbs)
    (OUT/'pr32-residual.json').write_text(json.dumps(d,indent=2)+'\n')
    print(us,d['effective_gbs_if_all_cold'],d['min_non_hbm_bytes_at_configured_peak'],observed['metrics']['step_medians'],observed['tile_count'])
if __name__=='__main__':review()
