"""Score three physical receipts against both predeclared candidate-order models."""
import json
from pathlib import Path

from calibration.msopprof import sha

BASE = Path(__file__).parent
FREEZES = dict(initial=('v113_frozen.json','cad6aecd937a30fcda1ad6339fd1f34a09fd7abcd1e1ea7c11c6f08a216dc2b5'),
               final=('v113b_frozen.json','902e5aa582e606e6e78dba9bb1ec1ef1fecf512d64e81705aa8040ab3a660dc5'))


def main():
    output = {'freeze_sha256':{name:h for name,(_,h) in FREEZES.items()},'rows':[]}
    for arm in ('shared','moving','rotated'):
        receipt=BASE/f'v113_control_{arm}.json'
        observed=json.loads(receipt.read_text())
        assert observed['all_ordered_lane_streams_match'] and observed['burst_counts_match_copy_pa_pair']
        ca=observed['device_us']
        row=dict(arm=arm,device_us=ca,physical_receipt=receipt.name,receipt_sha256=sha(receipt),
                 final_role='unblinded after final freeze' if arm=='moving' else 'mechanism selection replay')
        for version,(fn,h) in FREEZES.items():
            assert sha(BASE/fn)==h
            freeze=json.loads((BASE/fn).read_text())
            (r,)=(i for i in freeze['prices'] if i['case']=='control_'+arm and i['sku']==observed['sku'])
            assert r['candidate_order_overlap']==observed['logical_all_row_pair_same_rank'] and r['shape']==observed['shape']
            row[version]={k:{'us':r[k]['ca_window_us'],'signed_error_pct':100*(r[k]['ca_window_us']/ca-1)}
                          for k in ('shape_only','known_layout')}
        output['rows'].append(row)
    (BASE/'v113_control_scored.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
