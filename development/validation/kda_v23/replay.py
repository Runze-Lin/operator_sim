"""Apply development resource counters to the previously qualified DT control, without CA."""
import qualify
from series import BIN, DT, FIRST_PLAN, MEMORY, OUT, REG,probe,read,save


def main():
    reg=read(REG)
    old=FIRST_PLAN.parent
    prior=read(old/'verified.json')
    cost=read(old/'raw/cost.json')
    case=dict(id='reused-ascend950dt-t2-h1',sku=DT,physical_memory=MEMORY[DT],shape=[1,2,1,1,128,128],
              work_items=1,purpose='reused matched DT baseline; zero new CA',
              oracle=dict(attention=256,aqk=128,akk=128,state_vk=16384))
    result=qualify.verify(reg,case,old/'raw',cost)
    assert result['qualified'] and result['serialized_device_us']==prior['serialized_device_us']
    assert [r['aggregate_sha256'] for r in result['stages']]==[r['aggregate_sha256'] for r in prior['stages']]
    save(OUT/(case['id']+'.json'), dict(case=case,cost=cost,incremental_ca_seconds=0,original=probe.rel(old/'verified.json'),
        original_receipt_sha256=probe.digest(old/'verified.json'),registration_sha256=probe.digest(REG),**result))
    print(result['serialized_device_us'])

if __name__=='__main__':main()
