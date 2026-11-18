"""Qualify Post DB attribution by verifying the single BF16 output buffer byte-for-byte."""
import json
from pathlib import Path
import sqlite3
from recover import ROOT, OUT, sha256


def verify(row):
    with sqlite3.connect((ROOT / row['source']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        records = db.execute("""
            SELECT ExtendParams FROM ExecutedInstructions
            WHERE json_extract(ExtendParams,'$.dst_mem')='OUT'
        """)
        intervals=[]
        starts=[]
        for (text,) in records:
            p = json.loads(text)
            assert p['queue_type']=='MTE3' and p['src_mem']=='UB'
            address,length,bursts,step=(p[k] for k in ('dst_addr','burst_len','burst_number','dst_stride'))
            assert p['processed_bytes']==length*bursts and length>0 and bursts>0
            starts.append(address)
            intervals.extend((address+i*step,address+i*step+length) for i in range(bursts))
    t,n,d = row['shape']
    expected = 2*t*n*d
    intervals.sort()
    total = sum(end-start for start,end in intervals)
    union=0
    prev=0
    for start,end in intervals:
        union += max(0,end-max(start,prev))
        prev=max(end,prev)
    span = intervals[-1][1]-intervals[0][0] if intervals else 0
    return dict(shape=row['shape'], id=row['id'], source=row['source'], launch=row.get('launch'),
                db_sha256=row['db_sha256'], writes=len(starts), unique_starts=len(set(starts)),
                expected_bytes=expected, transfer_bytes=total, union_bytes=union, span_bytes=span,
                actual_base=hex(intervals[0][0]) if intervals else None,
                qualified=(expected==total==union==span))


def main():
    historical=[r for r in json.loads((ROOT/'results/v0.10/wide/rows.json').read_text()) if r['op']=='mhc_post']
    history=[]
    for r in historical:
        db=r['db_source']
        rec=dict(r,source=db,db_sha256=r.get('db_sha256') or sha256(ROOT/db))
        history.append(verify(rec))
    (OUT/'development-output-audit.json').write_text(json.dumps(history,indent=2)+'\n')
    original=json.loads((OUT/'original-attempt.json').read_text())
    good=[]
    audits=[]
    for r in original['completed']:
        a=verify(r)
        audits.append(a)
        if a['qualified']:
            good.append({**r,'output_audit':a})
    for path in sorted(OUT.glob('retry-v11_mhc_post_*.json')):
        for r in json.loads(path.read_text())['completed']:
            a=verify(r)
            audits.append(a)
            if a['qualified'] and r['role']=='validation':
                good.append({**r,'output_audit':a})
    (OUT/'output-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print([(r['shape'],r['qualified'],r['writes'],r['transfer_bytes'],r['union_bytes'],r['span_bytes']) for r in audits])
    cases=json.loads((OUT/'freeze.json').read_text())['cases']
    if len(good)==10:
        by_id={r['id']:r for r in good}
        assert len(by_id)==10
        (OUT/'mhc_post-0-10.json').write_text(json.dumps([by_id[c['id']] for c in cases],indent=2)+'\n')
    else:
        raise ValueError(f'{len(good)} target DBs qualify')


if __name__=='__main__':
    main()
