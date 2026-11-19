"""Compare repeated full descriptors (same payload) with the local issue price."""
from statistics import median
from large_cast import DT,PR,OUT,read,write


def terminal_intervals(data):
    rows=[]
    for case in data['cases']:
        t,h=case['shape']
        for chip in (DT,PR):
            if 'ca_us' not in case['devices'][chip]:continue
            prefix=f'cast_{t}_{h}_{chip}'
            audit=read(OUT/(prefix+'-dma.json'))
            diag=case['current'][chip]['diagnostics'];cap=diag['target_plan']['ub_tile_elements']
            prologue=diag['stages_us']['steady_tiles']
            local=diag['target_plan']['busy_core_elements']
            nominal=prologue*cap/(local-cap) if local>cap else None
            row=dict(shape=[t,h],sku=chip,terminal=audit['terminal'],model_full_steady_tile_us=nominal)
            for q,size in [('MTE2',2*cap),('MTE3',4*cap)]:
                ev=sorted((x for x in audit['terminal_dma'] if x['queue']==q and x['bytes']==size),key=lambda x:x['ts_us'])
                gaps=[b['ts_us']-a['ts_us'] for a,b in zip(ev,ev[1:])]
                row[q]=dict(bytes=size,observed=len(ev),duration_median_us=median(x['duration_us'] for x in ev) if ev else None,
                            issue_interval_median_us=median(gaps) if gaps else None)
            rows.append(row)
    write(OUT/'tile-intervals.json',rows)
    return rows
