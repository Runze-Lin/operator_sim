"""Experimental one-UB-block N=4 FP32, inference-only Sinkhorn issue schedule."""
from functools import lru_cache
from math import ceil
from pathlib import Path
from statistics import mean
from evidence import OUT,read

@lru_cache(maxsize=1)
def prices():return read(OUT/'prices.json')

def forecast(shape,iters,chip,previous=None):
    t,n=shape
    if type(t)!=int or t<1 or n!=4 or type(iters)!=int or not 1<=iters<=100:raise ValueError('FP32 inference N=4, T>0, 1..100 iterations')
    if previous is not None and (type(previous)!=int or not 0<=previous<=chip.vector_cores):raise ValueError('resident AIV outside target')
    p=prices();row=ceil(t/chip.vector_cores);active=ceil(t/row)
    # Both host candidates use two 256 B aligned register buffers and mask reserve.
    # Conservative backward split consumes 448 bytes per local 4x4 FP32 matrix.
    if 2*(row*(2*16+2*4*8+2*8)*4)+64+3*256>chip.details.get('ub_bytes',0):
        raise ValueError('Multiple UB iterations: no validated inter-block issue contract')
    groups=ceil(row/p['aligned_rows_per_vreg'])
    clock=chip.frequency_ghz*1000
    vector_price=chip.vector_tops['32']
    if not vector_price:raise ValueError('FP32 Vector throughput missing')
    rate=(vector_price*1e12/chip.vector_cores/(chip.frequency_ghz*1e9))/p['reference_vec_fp32_ops_per_cycle']
    payload_bytes=64*row; shared=64*t
    # Neither 950 SKU gives an identified per-AIV MTE bandwidth here. Use the
    # declared local price if available and keep a separate chip-wide byte floor.
    def dma_floor(byte_count,stage):
        rate=chip.details.get(stage+'_bandwidth_gbs')
        return byte_count*chip.frequency_ghz/rate if rate else 0.
    data_floor=max(shared*chip.frequency_ghz/chip.hbm_bandwidth_gbs,dma_floor(payload_bytes,'mte2'))
    write_floor=max(shared*chip.frequency_ghz/chip.hbm_bandwidth_gbs,dma_floor(payload_bytes,'mte3'))
    schedules={}
    for state,pr in p['phase_prices'].items():
        c={k:fixed+slope*active for k,(fixed,slope) in pr.items()}
        td=c['tiling_issue'];te=td+max(c['tiling_wait'],dma_floor(96,'mte2'))
        setup=te+c['setup_from_tile'];rd=setup+c['input_issue'];re=rd+max(c['input_wait'],data_floor)
        soft=re+c['soft_after_input']
        col=max(soft+p['softmax_col_floor']*groups/rate,setup+c['first_col_from_setup'])
        row0=col+p['first_col_to_row']*groups/rate
        first_pair=p['first_pair']*groups/rate;steady=p['steady_pair']*groups/rate
        scatter=row0+(first_pair+max(0,iters-2)*steady if iters>=2 else 0.)
        wd=scatter+c['write_gap']*groups/rate;we=wd+max(c['write_wait'],write_floor)
        schedules[state]=dict((k,v/clock) for k,v in {
            'tiling_begin':td,'tiling_end':te,'setup_vf':setup,'read_begin':rd,'read_end':re,'softmax_vf':soft,
            'first_col_vf':col,'first_row_vf':row0,'first_pair_period':first_pair,'steady_pair_period':steady,
            'first_row_to_col':p['row_col']*groups/rate,'steady_row_to_col':p['steady_row_col']*groups/rate,
            'scatter_vf':scatter,'write_begin':wd,'write_end':we,'end':we+c['end']}.items())
    chosen='first' if previous==0 else 'resident' if previous is not None and previous>=active else 'expand' if previous is not None else 'unknown'
    schedule=schedules[chosen] if chosen!='unknown' else {key:mean(s[key] for s in schedules.values()) for key in schedules['first']}
    return dict(latency_us=schedule['end'],selected=chosen,active_aiv=active,rows_per_core=row,register_passes=groups,
                vf_calls_per_core=2*iters+2,bytes_read=shared,bytes_written=shared,vf_setup_epochs=schedule,
                scenario_us={state:sc['end'] for state,sc in schedules.items()},extrapolated_local_grain=row>4)
