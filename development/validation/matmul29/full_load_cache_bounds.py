"""Secondary, named cold/warm ingress conditions; never replace the old center forecast."""
from datetime import datetime,timezone
import json
from full_load_freeze import ROOT,OUT,CHIPS,read,digest
from operator_model.chips import load_chip

def register():
    old=read(OUT/'freeze.json');rows=[]
    for r in old['rows']:
        if r['shape'][0]<262144:continue
        p=r['candidate'];chip=load_chip(r['chip']);cap=chip.details['l2_bytes'];a=p['work']['a_read_bytes']
        cs=p['cycles'];scale=chip.frequency_ghz*1000;span=min(cap,a)
        hbm=cs['chip_memory_finish']-span/p['rates']['hbm_bytes_per_chip_cycle']
        new=(max(hbm,cs['core_finish'],cs['l2_finish'])+cs['final_write'])/scale
        rows.append(dict(shape=r['shape'],sku=chip.name,original_center_us=p['latency_us'],up_to_one_l2_of_A_initially_warm_us=new,
                         warm_credit_bytes=span,all_output_writeback_cold_us=p['all_output_writeback_us'],
                         incremental_input_price_us_per_MiB=2**20/(1000*chip.hbm_bandwidth_gbs),
                         role='development or already scored' if r['shape'][0]==262144 else
                         'future only' if not r['acquisition'] else 'original receipt already visible' if (ROOT/r['acquisition']).exists() else 'remaining blind, named condition only'))
    with (OUT/'cache-conditions.json').open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),original_freeze_sha256=digest(OUT/'freeze.json'),
                       method='Same critical-AIC/L2 schedule, first fill, shared-bus price, pipeline drain and minimum C writeback. Compare zero input credit to credit=min(L2 capacity, A bytes); it is conditional on those bytes never going to HBM. Full remaining C output writeback shown separately. No fit or posterior mean.',rows=rows),f,indent=2);f.write('\n')
if __name__=='__main__':register()
