"""Qualify measured instruction windows and retain all blocks of the last AIV."""
import json
import math
import mmap
import re
from common import *
from development.validation.cast22.slot_figure import measured
from development.validation.vector21.figures import cast
from development.validation.prefill_timelines.compare import draw
from development.validation.sku19.collect import capture
from campaign_collect import stages


def endpoint(shard):
    snippets={}
    with shard.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
        for name in ('END','END_LABEL','DCI'):
            m=b'"name":"'+name.encode()+b'"'
            at=data.rfind(m) if name=='DCI' else data.find(m)
            if at<0:
                if name=='END':raise ValueError('missing END in '+str(shard))
                continue
            if name!='DCI' and data.find(m,at+len(m))>=0:raise ValueError('repeated END in '+str(shard))
            begin=data.rfind(b'{"args":{',max(0,at-32768),at)
            if begin<0:raise ValueError('malformed END')
            event=json.JSONDecoder().raw_decode(data[begin:at+2048].decode())[0]
            if event['name']!=name or event['ph']!='X':raise ValueError('malformed END record')
            if name=='DCI' and int(event['args']['pc_addr'],16)!=int(snippets['END']['pc'],16)-4:continue
            snippets[name]=dict(tick=round(event['ts']*1650),retire=round((event['ts']+event['dur'])*1650),pc=event['args']['pc_addr'])
    return snippets


def finish(job,folder):
    cost=read(folder/'cost.json')
    got,groups=stages(job,job['sku'],folder)
    stage,=got; t,h=job['shape'];log=(folder/'run.log').read_text(errors='replace')
    if stage['kernel']!=job['kernel'] or stage['binary_sha256']!=job['kernel_binary_sha256']:raise ValueError('actual kernel or device ELF changed')
    if f'case=cast_bf16_fp32 api=aclnnCast R={t} D={h}' not in log or groups!=[f'correctness count={t*h} bad=0']:raise ValueError('host input/golden count mismatch')
    agg=ROOT/stage['trace']; shards={p.parent.name:p for p in agg.parent.glob('core*/trace.json')}
    if set(stage['windows'])!=set(shards) or stage['active_aic'] or not shards:raise ValueError('roster mismatch')
    ends={c:endpoint(p) for c,p in shards.items()}
    # Removing the common time origin should change no per-core finishing offsets.
    origin=[max(m['retire'] for m in e.values())-1650*stage['windows'][c][1] for c,e in ends.items()]
    if max(origin)-min(origin)>3:raise ValueError('per-core final X label mismatch')
    return dict(qualified=True,case=job['id'],chip=job['sku'],shape=job['shape'],ca_us=stage['ca_us'],api=job['api'],
                correctness=groups,cost=cost,stage=stage,terminal_markers=ends,source='official msopprof per-instruction all-core window',
                model_us=job['forecasts'][job['sku']]['us'],projection_970dt_us=job['forecasts'][FUTURE]['us'],signed_pct=100*(job['forecasts'][job['sku']]['us']/stage['ca_us']-1),
                freeze_sha256=sha(REG),log_sha256=sha(folder/'run.log'))


def plot_ready(job,result,slug=None):
    # Verifies GM address continuity, UB slot reuse, strict MTE2/VCVT/MTE3
    # ordering and exact vector group counts. An incomplete mapping is never drawn.
    slug=slug or job['id']
    t,h=job['shape'];sku=job['sku']; actual=measured(ROOT,result)
    tiles=actual['tiles']
    dma={q:[dict(start=b[q]['window'][0],length=b[q]['window'][1]-b[q]['window'][0],bytes=b[q]['bytes']) for b in tiles] for q in ('MTE2','MTE3')}
    traces=dict(dma=dma,conversion_union=[(b['vcvt'][0],b['vcvt'][1]-b['vcvt'][0]) for b in tiles],terminal=actual['core'],ca_us=result['ca_us'],trace_sha256=actual['all_core_trace_sha256'])
    pred=job['forecasts'][sku];res=cast(job['shape'],sku,traces,frozen=pred)
    cap=res['plan']['ub_tile_elements'];n=len(tiles);busy=res['plan']['busy_core_elements']
    if actual['active_aiv']!=pred['active_aiv']:raise ValueError('active AIV count differs; retain raw receipt and block identities')
    d=dict(op='cast_bf16_fp32',model_name='DeepSeek V4 Pro',shape=job['shape'],batch=8,sequence=t//8,chip=sku,
           active_aiv=actual['active_aiv'],kernel=result['stage']['kernel'],model_us=pred['us'],ca_us=result['ca_us'],ca_core=actual['core'],
           ca_tile_elements=[b['n'] for b in tiles],model_tile_elements=[min(cap,busy-i*cap) for i in range(n)],
           actual_tiles_with_addresses=tiles,model_tiles=res['model'],ca_tiles=res['observed'],projection_970dt_us=job['forecasts'][FUTURE]['us'],
           source=actual['shard'],shard_sha256=actual['shard_sha256'],trace_sha256=actual['all_core_trace_sha256'],binary_sha256=actual['binary_sha256'],
           role='full instruction timing; coordinated direct ACLNN laboratory sample of the model-shaped cast',origin='earliest instruction on any active AIV',model_semantics=res['phase'])
    temp=OUT/(slug+'-plot.tmp.png');draw(d,temp);temp.replace(OUT/(slug+'-plot.png'));save(OUT/(slug+'-blocks.json'),d)
    return len(tiles)


if __name__=='__main__':
    # Offline compatibility controls can validate extraction on an existing official full window.
    sku=DT
    old=read(LEGACY/('cast_262144_64_'+sku+'.json'))
    old['api']='aclnnCast';old['chip']=sku
    reg=read(REG);j=dict(reg['jobs'][0],shape=old['shape'],sku=sku)
    from operator_model import estimate
    for target in (DT,PR,FUTURE):
        e=estimate('cast_bf16_fp32',old['shape'],chip=target,dtype='bf16');j['forecasts'][target]=dict(us=e.latency_us,active_aiv=e.active_aiv,diagnostics=e.projection['timing_diagnostics'])
    print(plot_ready(j,old,'archived-pro32k-dt-control'))
