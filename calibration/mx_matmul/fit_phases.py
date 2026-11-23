"""Recompute the 1.2 MX phase prices from local development receipts and raw traces.

Usage: python -m calibration.mx_matmul.fit_phases --freeze development-freeze.json
       --raw-root runs/mx12 --output mx_matmul_phases.json
The raw root holds native/<case-id>/ and mixed/<case-id>/. No CANN installation
or new CA run is required. The output must be a new file.
"""
import argparse
import json
from math import log2
from pathlib import Path
from statistics import median

from calibration.__main__ import digest, encoded, ingest, read, require, unseal
from calibration.msopprof import sha
from calibration.mx_matmul.native_phases import summarize
from operator_model import estimate, load_chip

ROUTES = {'matmul_mx_native': ('native', 'a4w4_native'), 'matmul_mx_mixed': ('mixed', 'a8w4_inline_dequant')}
SKUS = ('Ascend950DT_9582', 'Ascend950PR_957b')
SOURCE = ('CANN 9.2.1 verified full-window MX DT/PR traces, M=1,256 N=128 K=256 only. '
          'Terminal arrival is MMAD issue spread. Prices in device cycles. SKU fields are medians of '
          'two stage observations; pooled fields transfer the median of all four. '
          'Cold HBM traffic remains a capacity scenario.')


def simulator_directory(raw, stage):
    folders = sorted((raw/'profile').glob('OPPROF_*/simulator'))
    require(len(folders) == 1, f'Expected one CA kernel under {raw}')
    sim = folders[0]
    shards = {p.parent.name: p for p in sim.glob('core*.*core*/trace.json')}
    require(set(shards) == set(stage['launched_cores']), 'Incomplete set of physical-core shards')
    if 'physical_shard_sha256' in stage:
        hashes = {name: sha(p) for name,p in shards.items()}
        require(hashes == stage['physical_shard_sha256'] and digest(encoded(hashes)) == stage['trace_sha256'],
                'Physical traces differ from the development receipt')
    else:
        # The initial V3 collector fingerprinted the merged trace.
        require(sha(sim/'trace.json') == stage['trace_sha256'], 'Merged trace fingerprint differs')
    return sim


def weight_issue_us(sim, first_handoff_us, packed_bytes):
    reads, handoffs = [], []
    for shard in sim.glob('core*.veccore0/trace.json'):
        for event in read(shard)['traceEvents']:
            op = event.get('name')
            if event.get('ph') != 'X' or op not in ('MOV_SRC_TO_DST_ALIGNv2','MOV_UB_TO_L1'):
                continue
            info = json.loads(event['args']['detail'])
            if op == 'MOV_UB_TO_L1':
                handoffs.append(event['ts'])
            elif info['processed_bytes'] == packed_bytes:
                # The other MTE2 descriptor loads the 112-byte tiling record.
                reads.append(event['ts'])
    require(reads and handoffs, 'AIV0 payload read or UB-to-L1 timestamp missing')
    return first_handoff_us + min(reads) - min(handoffs)


def observation(raw, frozen, case):
    receipt = read(raw/'receipt.json')
    measured = ingest(frozen, receipt)
    require(measured['case_id'] == case['id'] and len(receipt['stages']) == 1, 'Mismatched case or kernel count')
    stage = receipt['stages'][0]
    ticks = stage['ticks_per_us']
    sim = simulator_directory(raw, stage)
    summary = summarize(sim)
    require(abs(summary['full_window_us']-measured['ca_us']) < .001, 'Phase clock differs from qualified CA window')
    r = estimate('matmul', case['shape'], case['attrs'], dtype=case['dtype'], chip=case['chip'])
    g, a = r.projection, r.active_aic
    plan = g['target_plan']
    # Only service terms derived from raw work, per-core ports and hardware
    # peaks are read here. Existing fitted clocks and predictions are unused.
    service, rates = g['page_service_cycles']['first'], g['prices']
    require(plan['pages_per_tile'] == 1 and summary['active_mmad_aics'] == a and a > 1,
            'Requires one verified K fragment and the observed active AIC count')
    m,n,k = case['shape']
    h,w = min(plan['tile'][0],m), min(plan['tile'][1],n)
    wt = max(2*h*w/rates['mte3_bytes_per_cycle_core'], 2*h*w*a/rates['l2_bytes_per_cycle_chip'])
    mixed = case['contract'] == 'matmul_mx_mixed'
    p, core = summary['phases'], summary['last_mmad_core_detail']
    first = (weight_issue_us(sim, p['MTE3']['first_issue_us'], w*k//2)
             if mixed else p['MTE2']['first_issue_us'])
    spread = (p['MMAD']['last_issue_us']-p['MMAD']['first_issue_us'])*ticks
    row = dict(chip=case['chip'], elements=w*k, launch_cycles=first*ticks,
               cube=(core['MMAD']['retire']-core['MMAD']['first'])*ticks-service['cube'],
               l0=(core['MMAD']['first']-core['MTE1']['first'])*ticks-service['load'],
               drain_cycles=(summary['full_window_us']-p['MMAD']['last_retire_us'])*ticks-wt,
               spread_cycles=spread/(a-1 if mixed else log2(a)),
               read_latency_cycles=(p['VECTOR' if mixed else 'MTE1']['first_issue_us']-first)*ticks-service['read'])
    if mixed:
        row['unpack_window'] = (p['MTE3']['first_issue_us']-p['VECTOR']['first_issue_us'])*ticks
        row['handoff'] = (p['MTE1']['first_issue_us']-p['MTE3']['first_issue_us'])*ticks-service['handoff']
    return row


def stage_prices(rows, mixed):
    def med(key, data=rows):
        return round(max(0, median(r[key] for r in data)), 2)
    parameters = dict(l0_latency_cycles=med('l0'), cube_latency_cycles=med('cube'))
    if mixed:
        # Determine intercept and slope independently on each 950 SKU, then
        # express the observed values/cycle as nominal INT8 issue equivalents.
        equivalents, setups = [], []
        for sku in SKUS:
            small, large = sorted((r for r in rows if r['chip']==sku), key=lambda r:r['elements'])
            slope = (large['unpack_window']-small['unpack_window'])/(large['elements']-small['elements'])
            require(slope > 0, 'Unpack service slope must be positive')
            chip = load_chip(sku)
            equivalents.append(slope*chip.vector_tops['8']*1000/(chip.frequency_ghz*chip.vector_cores))
            setups.append(small['unpack_window']-small['elements']*slope)
        parameters.update(unpack_peak_equivalents_per_weight=round(median(equivalents)),
                          unpack_setup_cycles=round(median(setups)), handoff_latency_cycles=med('handoff'))
    for sku in ('pooled',*SKUS):
        data = rows if sku=='pooled' else [r for r in rows if r['chip']==sku]
        parameters[sku] = {key: med(key,data) for key in
                           ('launch_cycles','read_latency_cycles','spread_cycles','drain_cycles')}
    return parameters


def fit(freeze_path, root):
    frozen = unseal(read(freeze_path), 'freeze')
    expected = {(route, chip, (m,128,256)) for route in ROUTES for chip in SKUS for m in (1,256)}
    cases = frozen['cases']
    require(len(cases)==len(expected) and {(c['contract'],c['chip'],tuple(c['shape'])) for c in cases}==expected,
            'This 1.2 phase estimator requires the eight development cases M=1/256 N=128 K=256')
    prices = {}
    for contract, (folder,key) in ROUTES.items():
        rows = [observation(root/folder/c['id'], frozen, c) for c in cases if c['contract']==contract]
        prices[key] = stage_prices(rows, contract=='matmul_mx_mixed')
    return {**prices, 'source': SOURCE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'raw-root', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    result = fit(args.freeze, args.raw_root)
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')


if __name__ == '__main__':
    main()
