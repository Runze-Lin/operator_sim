"""Reexecute the archived 239 real dispatch sequences through installed public streams."""
import json
from pathlib import Path
from hashlib import sha256
from math import isclose
from time import perf_counter_ns
import sys, tempfile


def ordered(api, root, save):
    from development.validation.post29 import replay
    from development.validation.post32.evaluate import old_api
    tmp=tempfile.TemporaryDirectory();old=old_api(tmp.name)
    source=json.loads((root/'results/v0.32/post/replay.json').read_bytes())
    class Dispatch:
        def __init__(self,sku):
            self.chip=sku;self.host=api.OperatorStream(sku);self.future=api.OperatorStream('Ascend970DT');self.base=old.OperatorStream(sku);self.visible=old.OperatorStream(sku,track_post_history=True)
        def forget(self):
            self.host.forget_history();self.future.forget_history();self.base.forget_history();self.visible.forget_history()
        def call(self,shape,op='mhc_post',attrs=None):
            start=perf_counter_ns();r=self.host.advance(op,shape,attrs);wall=(perf_counter_ns()-start)/1e6
            t=perf_counter_ns();f=self.future.advance(op,shape,attrs);future_wall=(perf_counter_ns()-t)/1e6
            base=self.base.advance(op,shape,attrs);visible=self.visible.advance(op,shape,attrs)
            if op!='mhc_post':return
            d=visible.projection['timing_diagnostics'];sc=d.get('ordered_scenario',d.get('context'))
            return dict(shape=shape,chip=self.chip,old_us=base.latency_us,ordered_us=visible.latency_us,scenario=sc,
                        target_plan=r.projection['target_plan'],stream_us=r.latency_us,future=f.latency_us,wall=wall,future_wall=future_wall)
    replay.Dispatch=Dispatch;replay.mixed_archive();replay.registered()
    original={x['sha256']:x for x in source['rows']}; assert original.keys()==replay.rows.keys()
    for k,r in replay.rows.items():
        ref=original[k];assert r['shape']==ref['shape'] and r['chip']==ref['chip'] and isclose(r['ca_us'],ref['ca_us'],abs_tol=.004)
    assert source['original_sources']==replay.source_hashes
    save('ordered-wheel-check.json',dict(historical_physical_windows=len(original),source_sha256=replay.source_hashes,
        archive='results/v0.32/post/replay.json',version=api.__version__,module=api.__file__,
        protocol='Fresh public OperatorStream streams on each chip plus an independent 970DT stream. Historical setup calls, unknown history, repeated calls and resets replayed from original source schedules, no per-case measurement passed to API.'))
    tmp.cleanup()
    return replay.rows
