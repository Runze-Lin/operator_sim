"""Pre-observation registration for two production-shaped contiguous casts."""
from math import ceil,prod
import gzip
from time import perf_counter
from common import *
from operator_model import estimate

SHAPES=[[8*131072,64],[8*262144,64]]
NAMES=['bf16_fp32_cast','cast_bf16_fp32']


def catalog():
    # Published whole-run receipts, independent vector ledger, raw names for started attempts.
    rows=[]
    for path in LEGACY.glob('cast_*_Ascend950*.json'):
        r=read(path)
        if r.get('qualified') and r.get('stage',{}).get('kernel','').startswith('Cast_'):
            rows.append((r['shape'],rel(path)))
    for path in [ROOT/'results/v0.19/workloads/cast-qualified.json',ROOT/'results/v0.21/vector/holdout.json']:
        for r in read(path):
            if r.get('qualified') and ('cast' in r.get('op','').lower() or 'cast' in r.get('id','').lower() or path.name=='cast-qualified.json'):
                rows.append((r['shape'],rel(path)))
    return rows


def register():
    if REG.exists():raise FileExistsError(REG)
    original=read(LEGACY/'pro-freeze.json')
    if sha(HOST)!=original['runner_sha256']:raise ValueError('Cast test driver changed')
    observed=catalog();jobs=[]
    for shape in SHAPES:
        t,h=shape;n=t*h
        contract_file=ROOT/'results/v0.20/workloads/contracts.jsonl.gz'
        scenario=dict(model='DeepSeek-V4-Pro',seq=t//8,batch=8,tp=8,dp=1,ep=8,chunk=0,step=0,ep_rank=0)
        with gzip.open(contract_file,'rt') as handle:
            matched=[d for l in handle if (d:=json.loads(l))['scenario']==scenario]
        [scenario_row]=matched
        [node]=[node for node in scenario_row['contracts'] if node['op']=='indexer.prepare_dsa_weights.float']
        if node['standalone_call']!={'op':'cast_bf16_fp32','shape':shape,'attrs':{}}:raise ValueError('model-derived Cast contract changed')
        if any(sh==shape for sh,_ in observed):raise ValueError('existing shape in CA ledger')
        if any((ROOT/'runs/prefill20').glob(f'cast_{t}_{h}_*')):raise ValueError('earlier raw launch')
        flat=[dict(shape=sh,receipt=receipt) for sh,receipt in observed if prod(sh)==n]
        forecasts={}
        for sku in [DT,PR,FUTURE]:
            start=perf_counter();e=estimate('cast_bf16_fp32',shape,chip=sku,dtype='bf16',timing_mode='resource')
            forecasts[sku]=dict(us=e.latency_us,active_aiv=e.active_aiv,diagnostics=e.projection['timing_diagnostics'],
                model_query_ms=(perf_counter()-start)*1000,scope='scenario extrapolation' if sku==FUTURE else 'frozen pre-observation forecast')
        for sku in (DT,PR):
            old_path=LEGACY/('cast_2097152_32_'+sku+'.json');old=read(old_path);cost=old['cost']
            ratio=n/(2097152*32)
            # Peak in the control is a lower bound. 50% + 1 GiB headroom supports
            # the temporary overlap between vendor raw shards and converted JSON.
            gib=ceil(1.50*ratio*cost['peak_bytes']/2**30+1);hours=ceil(2.0*ratio*cost['seconds']/3600+0.35)
            id=f"pro-b8-{t//(8*1024)}ki-"+('dt' if sku==DT else 'pr')
            jobs.append(dict(id=id,op='cast_bf16_fp32',api='aclnnCast',dtype='bf16',output_dtype='fp32',shape=shape,batch=8,sequence=t//8,
               scenario=scenario,model_contract=node,sku=sku,runner=rel(HOST),args=list(map(str,shape)),kernels=1,nchecks=1,wall_s=3600*hours,
               raw_mib=1024*gib,logical_read_B=2*n,logical_write_B=4*n,golden_count=n,forecasts=forecasts,
               prior_equal_flattened=flat,prior_exact_shape=False,kernel=old['stage']['kernel'],kernel_binary_sha256=old['stage']['binary_sha256'],
               cost_reference=dict(receipt=rel(old_path),sha256=sha(old_path),seconds=cost['seconds'],peak_bytes_lower_bound=cost['peak_bytes'],retained=cost['raw_bytes'],element_ratio=ratio)))
    work=[contract_file,*HERE.glob('*.py'),HOST,ROOT/'development/validation/large20/cast.cpp',ROOT/'smoke/large_vector.cpp',
      ROOT/'development/validation/sku19/collect.py',ROOT/'development/validation/sku17/campaign_collect.py',ROOT/'development/validation/cast22/slot_figure.py',
      ROOT/'development/validation/prefill_timelines/compare.py',ROOT/'development/validation/vector21/figures.py',
      ROOT/'operator_model/cast_resource.py',ROOT/'operator_model/rms_resource.py',ROOT/'operator_model/chips.py',ROOT/'operator_model/vector_memory.py',
      ROOT/'scripts/cann-run.sh',LEGACY/'pro-freeze.json',ROOT/'results/cast-capacity/two-slot-freeze.json',*(ROOT/f'configs/chips/{s}.json' for s in (DT,PR,FUTURE))]
    save(REG,dict(utc=when(),jobs=jobs,hashes={rel(p):sha(p) for p in work},model_role='v0.23 production Cast, fully frozen, no fit on either requested matrix shape; identical flattened size to Flash where flagged.',
      actual_model_route=dict(source_revision=original['source_revision'],source=original['source'],input_units='tokens per rank=8 × unchunked context, replicated local indexer projection output width=64'),
      gates='All elements bit-exact, one real Cast ELF, correct simulated hardware runtime + DRAM, completed public official per-core window; actual last-core tiling and conversion/grant intervals separate.',
      concurrency=1,queue='After the existing small KDA sequence; while full-hidden Kimi is computing, only start after two minutes with 1m load <= 50% affinity count and available RAM >= 2 × case disk cap. Single CA, disk and CPU checked between jobs.',
      native_scope='If msopprof exceeds capacity after actual completion, preserve same-run native per-core window separately when every real start/done record and every numeric output can be proved. A missing instruction or native window remains N/A. No second native launch.'))
    print(sha(REG))
if __name__=='__main__':register()
