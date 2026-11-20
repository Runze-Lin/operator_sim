"""Translate saturated physical-channel payload into a fraction of configured HBM bandwidth."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35/quant'
src=ROOT/'results/v0.27/quant-channel-service.json'
row,=[r for r in json.loads(src.read_bytes())['rows'] if r['case']=='Kimi quant']
v=row['channel_1us_bins']['mixed/queued_95pct']['summed_channel_counts']
duty=(v['read']+v['write'])/v['slots']
spec=json.loads((ROOT/'configs/chips/Ascend950DT_9582.json').read_bytes())
wire=128*32  # one 32-byte beat per 1 ns period per 950DT channel (GB/s)
fraction=duty*wire/spec['hbm_bandwidth_gbs']
doc=dict(reference='Ascend950DT_9582',source_path=str(src.relative_to(ROOT)),source_sha256=sha256(src.read_bytes()).hexdigest(),
         method='Count effective 32-byte transfers in both directions over 1-us physical channel periods with mixed traffic and at least 95% queue occupancy.',
         read_beats=v['read'],write_beats=v['write'],channel_cycles=v['slots'],periods=v['slots']//1000,
         channel_count=128,dfi_ns=1.,bytes_per_beat=32,wire_gbs=wire,reference_hbm_bandwidth_gbs=spec['hbm_bandwidth_gbs'],
         physical_duty=duty,quant_mixed_bandwidth_efficiency=fraction)
(OUT/'service.json').write_text(json.dumps(doc,indent=2)+'\n')
