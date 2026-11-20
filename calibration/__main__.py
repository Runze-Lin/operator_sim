import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
FUTURE = 'Ascend970DT'
SCHEMA = 1
CASE_FIELDS = ('id', 'op', 'contract', 'chip', 'shape', 'attrs', 'dtype', 'api', 'outputs', 'prefix')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes(), parse_constant=lambda x: invalid_number(x))


def invalid_number(value):
    raise ValueError(f'Non-finite number: {value}')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(data):
    return json.dumps(data, allow_nan=False, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def seal(data):
    return {**data, 'sha256': digest(encoded(data))}


def unseal(data, kind):
    check = dict(data)
    fingerprint = check.pop('sha256', None)
    require(fingerprint == digest(encoded(check)), 'Document fingerprint mismatch')
    require(data['kind'] == kind and data['schema'] == SCHEMA, 'Unexpected document type/schema')
    return data


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(dt.tzinfo is not None, 'Timestamp needs an explicit timezone')
    return dt


def sha_field(s):
    return isinstance(s, str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)


def integer(n, minimum=0):
    return type(n) is int and n >= minimum


def valid_outputs(outputs):
    return (isinstance(outputs, dict) and bool(outputs)
            and all(isinstance(k,str) and k and integer(v,1) for k,v in outputs.items()))


def case_key(row):
    return tuple(encoded(row.get(k)) for k in CASE_FIELDS)


def predict(api, case, chip):
    common = dict(attrs=case['attrs'], dtype=case['dtype'])
    if case['prefix'] is None:
        return api.estimate(case['op'], case['shape'], chip=chip, **common)
    stream = api.OperatorStream(chip)
    for call in case['prefix']:
        if call == {'forget': True}:
            stream.forget_history()
        else:
            stream.advance(call['op'], call['shape'], call.get('attrs',{}), dtype=call.get('dtype'))
    return stream.advance(case['op'], case['shape'], **common)


def freeze(manifest, chip_files=()):
    import operator_model as api
    require(manifest['schema']==SCHEMA and isinstance(manifest['cases'],list) and manifest['cases'], 'Empty or unsupported case manifest')
    custom = {}
    for path in chip_files:
        config = api.load_chip(path)
        require(config.name not in custom, 'Duplicate chip configuration')
        custom[config.name] = config
    rows, chips = [], {}
    def describe(chip):
        return {name: dict(value) if hasattr(value,'items') else value
                for name,value in vars(chip).items()}
    for item in manifest['cases']:
        c = dict(attrs={}, dtype=None, prefix=None, contract=item['op'])
        c.update(item)
        require(set(c)==set(CASE_FIELDS), 'Case fields differ from documented schema')
        require(isinstance(c['id'],str) and c['id'] and isinstance(c['op'],str) and c['op'], 'Invalid id/op')
        require(isinstance(c['chip'],str) and c['chip'] and isinstance(c['api'],str) and c['api'], 'Supply measured SKU and native API')
        require(isinstance(c['shape'],list) and c['shape'] and all(integer(n,1) for n in c['shape']), 'Invalid shape')
        require(isinstance(c['attrs'],dict) and (c['prefix'] is None or isinstance(c['prefix'],list)), 'Invalid attrs/prefix')
        require(valid_outputs(c['outputs']), 'Declare every expected output and its full element count')
        chip = custom.get(c['chip']) or api.load_chip(c['chip'])
        chips[chip.name] = describe(chip)
        require(chip.name==c['chip'], 'A calibration SKU must name the actual measured device')
        start = perf_counter_ns()
        price = predict(api,c,chip)
        elapsed = (perf_counter_ns()-start)/1e6
        future_chip = custom.get(FUTURE) or api.load_chip(FUTURE)
        chips[future_chip.name] = describe(future_chip)
        future = predict(api,c,future_chip)
        require(price.latency_us is not None and price.latency_us>0, 'Input has no priced model')
        rows.append({**c,'predicted_us':price.latency_us,'query_ms':elapsed,
                     'extrapolated_970dt_us':future.latency_us})
    require(len({r['id'] for r in rows})==len(rows), 'Duplicate case id')
    # This fingerprint identifies the published code and prices used by this installation.
    package = Path(api.__file__).resolve().parent
    members = {str(p.relative_to(package)):digest(p.read_bytes()) for p in sorted(package.rglob('*'))
               if p.is_file() and p.suffix in {'.py','.json'}}
    return seal(dict(kind='freeze', schema=SCHEMA, model_version=api.__version__,
                     package_sha256=digest(encoded(members)), chips=chips, created_utc=datetime.now(timezone.utc).isoformat(), cases=rows))


def stage_window(s):
    for k in ('trace_sha256','kernel_binary_sha256'):
        require(sha_field(s[k]), 'Missing stage fingerprint '+k)
    require(isinstance(s['kernel'], str) and bool(s['kernel']), 'Missing runtime kernel name')
    clocks = s['ticks_per_us']
    require(type(clocks) in (int,float) and math.isfinite(clocks) and clocks>0, 'Invalid tick frequency')
    roster, cores = s['launched_cores'], s['cores']
    require(isinstance(roster,list) and roster and len(roster)==len(set(roster))
            and len(cores)==len(roster) and set(cores)==set(roster), 'Incomplete or duplicate core roster')
    for core in cores.values():
        lo, end, hi = (core[k] for k in ('first_tick','end_tick','retire_tick'))
        require(all(integer(t) for t in (lo,end,hi)) and lo<=end<=hi and lo<hi and integer(core['end_count'],1) and core['end_count']==1,
                'Invalid physical END, start or retirement')
    return (max(c['retire_tick'] for c in cores.values())-min(c['first_tick'] for c in cores.values()))/clocks


def ingest(frozen, receipt):
    unseal(frozen,'freeze')
    require(receipt.get('schema')==SCHEMA and receipt.get('kind')=='ca-receipt', 'Unexpected CA receipt type')
    require(receipt['timing']=='all-core-instruction' and receipt['complete'] is True, 'Require complete instruction timing')
    case = next((c for c in frozen['cases'] if c['id']==receipt['case']['id']), None)
    require(case and case_key(case)==case_key(receipt['case']), 'Receipt input differs from frozen input')
    begin, end = stamp(receipt['started_utc']), stamp(receipt['finished_utc'])
    require(begin<=end and end<=datetime.now(timezone.utc), 'Invalid acquisition dates')
    checks = receipt['correctness']
    require(set(checks)==set(case['outputs']) and all(
        integer(checks[k]['checked'],1) and checks[k]['checked']==n and integer(checks[k]['bad']) and checks[k]['bad']==0
        for k,n in case['outputs'].items()), 'Missing, partial or failing numerical result')
    require(isinstance(receipt['provider'],str) and receipt['provider'] and sha_field(receipt['run_log_sha256']),
            'Need provider and host log identity')
    stages = receipt['stages']
    require(isinstance(stages,list) and stages, 'No kernel stages')
    measured = sum(stage_window(s) for s in stages)
    ids = [s['trace_sha256'] for s in stages]
    require(len(ids)==len(set(ids)),'Repeated stage identity')
    binding = receipt.get('freeze_sha256')
    if binding is not None:
        require(binding==frozen['sha256'], 'Receipt belongs to a different freeze')
    role='replay'
    if begin>stamp(frozen['created_utc']):
        require(binding==frozen['sha256'], 'Prospective acquisition must record the freeze fingerprint')
        role='prospective'
    return seal(dict(schema=SCHEMA,kind='measurement',freeze_sha256=frozen['sha256'],case_id=case['id'],
                     identity=ids,kernel=[s['kernel'] for s in stages],provider=receipt['provider'],
                     receipt_sha256=digest(encoded(receipt)),data_role=role,ca_us=measured))


def metrics(rows):
    if not rows:
        return {'windows':0,'shapes':0,'wape_pct':None,'max_pct':None}
    return dict(windows=len(rows),shapes=len({tuple(r['shape']) for r in rows}),
                wape_pct=100*sum(abs(r['predicted_us']-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                max_pct=max(abs(r['signed_error_pct']) for r in rows))


def score(frozen, measurements):
    unseal(frozen,'freeze')
    cases = {c['id']:c for c in frozen['cases']}
    groups, rows, case_ids, ids = {}, [], set(), set()
    for m in measurements:
        unseal(m,'measurement')
        require(m['freeze_sha256']==frozen['sha256'] and m['case_id'] in cases, 'Wrong freeze binding')
        c = cases[m['case_id']]
        require(c['id'] not in case_ids and not ids.intersection(m['identity']), 'Duplicate case or physical stage')
        require(type(m['ca_us']) in (int,float) and math.isfinite(m['ca_us']) and m['ca_us']>0,'Invalid measurement')
        case_ids.add(c['id']); ids.update(m['identity'])
        row = dict(case_id=c['id'],op=c['op'],contract=c['contract'],chip=c['chip'],shape=c['shape'],attrs=c['attrs'],dtype=c['dtype'],
                   api=c['api'],kernel=m['kernel'],identity=m['identity'],data_role=m['data_role'],ca_us=m['ca_us'],
                   predicted_us=c['predicted_us'],extrapolated_970dt_us=c['extrapolated_970dt_us'],
                   signed_error_pct=100*(c['predicted_us']/m['ca_us']-1))
        rows.append(row)
        groups.setdefault((c['contract'],c['chip'],m['data_role']),[]).append(row)
    return dict(rows=rows,pending=sorted(cases.keys()-case_ids),summary=[
        dict(contract=k[0],chip=k[1],data_role=k[2],**metrics(group)) for k, group in sorted(groups.items())])


def main():
    cli=argparse.ArgumentParser(description='Freeze operator-sim predictions, import complete CA receipts, and score matching cases')
    commands=cli.add_subparsers(dest='cmd',required=True)
    for name,fields in [('freeze',('manifest','output')),('import',('freeze','receipt','output')),('score',('freeze',))]:
        cmd=commands.add_parser(name)
        for field in fields:cmd.add_argument(field,type=Path)
        if name=='freeze':cmd.add_argument('--chip-config', action='append', type=Path, default=[])
        if name=='score':cmd.add_argument('measurements',nargs='+',type=Path)
    args=cli.parse_args()
    try:
        if args.cmd=='freeze':doc=freeze(read(args.manifest),args.chip_config)
        elif args.cmd=='import':doc=ingest(read(args.freeze),read(args.receipt))
        else:doc=score(read(args.freeze),[read(p) for p in args.measurements])
        result=json.dumps(doc,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        if args.cmd=='score':sys.stdout.write(result)
        else:
            with args.output.open('x') as file:file.write(result)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        cli.exit(1,str(exc)+'\n')


if __name__=='__main__':
    main()
