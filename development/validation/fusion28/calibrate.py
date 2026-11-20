"""Measure increment prices only from DT instruction records, no total latency target."""
from pathlib import Path
from statistics import median
from landmarks import read,save,sha,OUT,ROOT,BASE

def main():
    ds=[r for r in read(OUT/'landmarks.json') if r['sku']=='Ascend950DT_9582']
    def case(t):
        return next(x for x in ds if x['shape'][0]==t and x['role'].startswith('v0.26'))
    one,two,many=case(449),case(513),case(4097)
    m1,m2,m9=[r['milestones'] for r in (one,two,many)];g=many['groups']
    data=read(ROOT/many['trace'])['traceEvents']
    scat=sorted((e for e in data if e.get('ph')=='X' and e['name']=='RV_VSCATTER'),key=lambda e:e['ts'])
    assert len({x['args']['pc_addr'] for x in scat})==1 and len(scat)==8*g
    ticks=[round(e['ts']*1650) for e in scat[::8]]
    orig=read(BASE/'prices.json')
    # v26 one-group rates and marginal cost from repeated executions of the
    # same helper; this keeps the fixed loop/queue issue latency separate.
    step=median([(m2['steady']-m1['steady'])/(two['groups']-1),(m9['steady']-m1['steady'])/(g-1)])
    delta=(m9['soft_duration']-m2['soft_duration'])/(g-2)
    soft_lead=m2['soft_duration']-(m2['first_col_issue']-m2['soft_issue'])
    two_tail=m2['steady']-median(m2['row_to_col'][1:])
    large_tail=m9['steady']-median(m9['row_to_col'][1:])
    prices=dict(phase_source=orig['phase_prices'],soft_one_floor=orig['softmax_col_floor'],soft_extra_cycles=delta,
        soft_start_to_first_col_linear_intercept=m2['soft_duration']-delta-soft_lead,
        col_row_second_pass=m2['first_col_to_first_row'],col_row_increment=(m9['first_col_to_first_row']-m2['first_col_to_first_row'])/(g-2),
        first_pair=orig['first_pair'],steady_pair=orig['steady_pair'],row_col_one=orig['row_col'],extra_pair_cycles=step,
        column_tail_two=two_tail,column_tail_inc=(large_tail-two_tail)/(g-2),scatter_additional=median(b-a for a,b in zip(ticks[2:],ticks[3:])),
        source=[dict(shape=r['shape'],trace=r['trace'],sha256=r['trace_sha256']) for r in (one,two,many)],
        source_kernel='MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0',
        scatter_pc=scat[0]['args']['pc_addr'],scatter_pass_ticks=ticks,soft_descriptor_overlap_cycles=soft_lead,
        previous_prices_sha256=sha(BASE/'prices.json'),landmarks_sha256=sha(OUT/'landmarks.json'))
    save(OUT/'prices.json',prices)
if __name__=='__main__':main()
