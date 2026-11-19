"""Compare sync fault counts at each ELF offset for the identical small Kimi case."""
from investigate import OUT, read, save

def pair():
    ids=['dt-t2-h1-varlen','pr-t2-h1-varlen']
    sync=[read(OUT/(c+'-sync.json')) for c in ids]
    nums=[read(OUT/(c+'-arrays.json'))['outputs'] for c in ids]
    if sync[0]['captured_text_sha256']!=sync[1]['captured_text_sha256']:
        raise ValueError('ELF text differs: offsets are not directly comparable')
    bins=[{f['offset']:f['occurrences'] for f in s['faults']} for s in sync]
    offsets=sorted(set(bins[0])|set(bins[1]))
    per_offset={k:[m.get(k,0) for m in bins] for k in offsets}
    result=dict(shape=[1,2,1,1,128,128],order=['DT','PR'],same_device_text_sha256=sync[0]['captured_text_sha256'],
        core_roster_aic=[len(s['aic']) for s in sync],no_mmad_aic=[sum(r['mmad_count']==0 for r in s['aic'].values()) for s in sync],
        identical_fault_offset_set=bins[0].keys()==bins[1].keys(),same_offsets=len(set(bins[0])&set(bins[1])),
        per_offset_different_counts={k:v for k,v in per_offset.items() if v[0]!=v[1]},
        per_offset_all_counts=per_offset,actual_output_pair_sha256={k:[n[k]['actual_sha256'] for n in nums] for k in nums[0]},
        raw_actual_bitwise_identical={k:nums[0][k]['actual_sha256']==nums[1][k]['actual_sha256'] for k in nums[0]})
    save(OUT/'paired-varlen.json',result)
if __name__=='__main__':pair()
