"""Join later qualified receipts to immutable prices without launching CA."""
from datetime import datetime, timezone
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.32/post'
def read(f):return json.loads(f.read_bytes())
def main():
    frozen=read(OUT/'freeze.json')
    split=read(ROOT/'results/v0.30/post/forward-score.json')['qualified']
    isolated=read(ROOT/'results/v0.31/post/score.json')['rows']
    channel=read(ROOT/'results/v0.30/post/channel-score.json')['new']
    extra=read(OUT/'channel-freeze.json')
    rows=[]
    def result(p,ca,source,scope):
        return dict(chip=p['chip'],shape=p['shape'],role=scope,ca_us=ca,receipt=source,
                    frozen=p,shape_signed_pct=100*(p.get('shape_us',p.get('candidate_shape_us'))/ca-1) if ca else None,
                    sequence_signed_pct=100*(p.get('ordered_us',p.get('candidate_repeat_us'))/ca-1) if ca else None)
    for p in frozen['predictions']:
        if p['chip']=='Ascend970DT' or p['data_role']!='target':continue
        source=None;ca=None
        if p['group']=='isolated_large':
            measured=next(r for r in isolated if r['op']=='mhc_post' and r['shape']==p['shape'] and r['chip']==p['chip'])
            if measured['ca_us'] is not None:
                assert measured['stage']=='qualified'
                source='results/v0.31/post/'+measured['receipt']
                ca=measured['ca_us']
        else:
            measured=next((r for r in split if r['chip']==p['chip'] and r['shape']==p['shape'] and r['ordinal']==p['ordinal']),None)
            if measured:
                source='results/v0.29/post/split-'+p['chip']+'.json'
                ca=measured['ca_us']
        rows.append(result(p,ca,source,p['group']))
    for p in extra['forward_rows']:
        if p['chip']=='Ascend970DT':continue
        measured=next((r for r in channel if r['chip']==p['chip'] and r['shape']==p['shape']),None)
        ca=measured['ca_us'] if measured else None
        file='results/v0.30/post/channel-'+p['chip']+'.json' if measured else None
        rows.append(result(p,ca,file,'candidate_channel_repeat'))
    (OUT/'forward.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        qualified=sum(r['ca_us'] is not None for r in rows),expected=len(rows)),indent=2)+'\n')

if __name__=='__main__':main()
