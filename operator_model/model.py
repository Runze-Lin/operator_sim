"""Shape-only tile/stage surrogate. Costs are CA calibrated, not ISA peaks."""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path


class FrozenDict(dict):
    """JSON-serializable read-only result, safe to share on cache hits."""
    def _readonly(self, *args, **kwargs):
        raise TypeError('Prediction results are read-only')

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict({k:freeze(v) for k,v in value.items()})
    if isinstance(value, (tuple,list)):
        return tuple(freeze(v) for v in value)
    return value


def ceildiv(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceildiv(a, b) * b


@dataclass(frozen=True)
class Case:
    op: str
    shape: tuple

    def __post_init__(self):
        object.__setattr__(self, 'shape', tuple(self.shape))
        required = 2 if self.op == 'quant' else 3
        if self.op not in ('quant', 'matmul', 'fa') or len(self.shape) != required:
            raise ValueError('Expected quant(rows,cols), matmul(M,N,K), or fa(Q,KV,D)')
        if any(type(d) is not int or not 1 <= d <= 4096 for d in self.shape):
            raise ValueError('Dimensions must be integers in [1,4096]')
        if self.op == 'quant' and self.shape[1] % 64:
            raise ValueError('v0.1 only supports quant cols divisible by 64')
        if self.op == 'fa' and (self.shape[2] != 64 or self.shape[0] > 128 or self.shape[1] > 1024):
            raise ValueError('v0.1 FA domain: B=H=1, D=64, Q<=128, KV<=1024, noncausal BF16 SBH')


def quant_partition(rows, cols, capacity=64):
    """Documented tail-axis block32 2D split, independently expressed in Python.

    Source: ops-nn arch35 DynamicMxQuantTailAxisTiling (see evidence-v1.md).
    These are source-derived hypotheses until checked against the packaged kernel.
    """
    width = 256 if rows * ceildiv(cols, 512) <= capacity * .75 else 512
    columns = ceildiv(cols, width)
    cores = min(capacity, rows * columns)
    divisors = {v for m in range(1, math.ceil(math.sqrt(cores) + 1)) for v in (m, cores // m)}
    options = []
    for rsplit in sorted(divisors):
        csplit = max(1, cores // rsplit)
        if rsplit > rows or csplit > columns:
            continue
        nr, nc = ceildiv(rows, rsplit), ceildiv(columns, csplit)
        hole = nr * nc
        if rsplit * csplit == cores:
            if rows % rsplit == 0 and columns % csplit == 0:
                hole = 0
            elif rows % rsplit == 0:
                hole -= nr * (columns % nc)
            elif columns % csplit == 0:
                hole -= (rows % nr) * nc
            else:
                hole -= (rows % nr) * (columns % nc)
        options.append((hole, csplit, rsplit, nr, nc))
    _, _, _, nr, nc = min(options)
    return {'active_aiv': ceildiv(rows, nr) * ceildiv(columns, nc),
            'rows_per_core': nr, 'cols_per_core': min(cols, nc * width), 'slice_cols': width}


def plan(case):
    """Return bounded structural work, never a measured duration or trace field."""
    s = case.shape
    if case.op == 'quant':
        rows, cols = s
        split = quant_partition(rows, cols)
        elems = split['rows_per_core'] * align(split['cols_per_core'], 256)
        units = {'read': 2 * elems / 1024, 'vector': elems / 256,
                 'write': (elems + elems / 32) / 1024, 'cube': 0., 'fix': 0.}
        return {**split, 'active_aic': 0, 'waves': 1, 'iterations': 1,
                'units': units, 'work_per_core': elems, 'useful_work': rows * cols,
                'logical_bytes': rows * cols * (3 + 1 / 32),
                'mapping': 'source-derived tail-axis block32 two-dimensional split'}
    if case.op == 'matmul':
        m, n, k = s
        # Deliberately bounded first model: elementary 16x64 output work units.
        # Compare predicted versus observed active cores; do not claim this is
        # a complete port of CANN's many tiling dispatch branches.
        tm, tn, tk = min(align(m, 16), 16), min(align(n, 16), 64), align(k, 16)
        tiles = ceildiv(m, tm) * ceildiv(n, tn)
        active = min(32, tiles)
        waves = ceildiv(tiles, active)
        units = {'read': 2 * (tm * tk + tn * tk) / 1024,
                 'cube': 2 * tm * tn * tk / 131072,
                 'fix': 2 * tm * tn / 1024, 'write': 0., 'vector': 0.}
        return {'active_aic': active, 'active_aiv': 0, 'waves': waves, 'iterations': 1,
                'tile': [tm, tn, tk], 'units': units, 'work_per_core': 2 * tm * tn * tk * waves,
                'useful_work': 2 * m * n * k, 'logical_bytes': 2 * (m*k + k*n + m*n),
                'mapping': '16x64 elementary-work tiling hypothesis; compare observed mapping'}
    q, kv, d = s
    tq, tkv = min(align(q, 16), 64), min(align(kv, 16), 128)
    qtiles, kvtiles = ceildiv(q, 64), ceildiv(kv, 128)
    units = {'read': 2 * (tq*d + 2*tkv*d) / 1024,
             'cube': 4 * tq * tkv * d / 131072,
             'vector': (tq * tkv / 2) / 256,
             'fix': 4 * tq * tkv / 1024, 'write': 2 * tq * d / 1024}
    return {'active_aic': min(32, qtiles), 'active_aiv': 2 * min(32, qtiles),
            'waves': ceildiv(qtiles, 32), 'iterations': kvtiles,
            'tile': [tq, tkv, d], 'units': units, 'work_per_core': 4 * tq * tkv * d * kvtiles,
            'useful_work': 4 * q * kv * d, 'logical_bytes': 2 * (2*q*d + 2*kv*d),
            'mapping': 'source-informed Q64/KV128 mixed pipeline; no split-KV or sparse attention'}


def schedule(op, p, stages):
    """Finite pipeline fill/steady/drain; phase services, not per-instruction replay."""
    read, vector, cube, fix, write = (stages[k] for k in ('read', 'vector', 'cube', 'fix', 'write'))
    if op == 'fa':
        # QK -> softmax -> PV, with shared Cube resource; running softmax/
        # accumulation creates a V+PV recurrence. MTE can overlap next tile.
        fill = read + cube + vector + fix
        ii = max(read, cube + fix, vector + cube / 2)
        return p['waves'] * (fill + (p['iterations'] - 1) * ii + write)
    fill = read + vector + cube + fix + write
    ii = max(read, vector, cube, fix, write)
    return fill + (p['waves'] - 1) * ii


def fit_nonnegative_line(xs, ys):
    """Two-parameter NNLS, solved by interior plus both boundary candidates."""
    if not xs or len(xs) != len(ys):
        raise ValueError('Empty or mismatched fit data')
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    var = sum((x-mx)**2 for x in xs)
    candidates = [(max(0., my), 0.)]
    norm = sum(x*x for x in xs)
    if norm:
        candidates.append((0., max(0., sum(x*y for x,y in zip(xs,ys))/norm)))
    if var > 1e-15:
        b = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var
        a = my - b*mx
        if a >= 0 and b >= 0:
            candidates.append((a,b))
    a,b = min(candidates, key=lambda c: sum((c[0]+c[1]*x-y)**2 for x,y in zip(xs,ys)))
    return {'intercept_us': a, 'slope_us_per_unit': b}


def line(cost, x):
    return cost['intercept_us'] + cost['slope_us_per_unit'] * x


def measured_stages(record):
    cores = record['measurement']['cores']
    def max_pipe(pipe, kind=None):
        return max((c['busy_union_us_by_pipe'].get(pipe, 0.) for c in cores if kind is None or c['kind']==kind), default=0.)
    return {'read': max_pipe('MTE2'), 'vector': max_pipe('VECTOR_FAMILY', 'AIV'),
            'cube': max_pipe('CUBE', 'AIC'), 'fix': max_pipe('FIXP', 'AIC'), 'write': max_pipe('MTE3', 'AIV')}


def fit(records, manifest_sha256):
    """Caller must supply calibration only. Reject leakage rather than filtering."""
    if any(r['case']['split'] != 'calibration' for r in records):
        raise ValueError('Holdout data must never enter fitting')
    identities = {json.dumps(r.get('identity'),sort_keys=True) for r in records}
    if len(identities) != 1:
        raise ValueError('Calibration software/hardware identities differ')
    result = {'schema_version': 1, 'version': '0.1.0', 'soc_version': 'Ascend950DT_9582',
              'manifest_sha256': manifest_sha256, 'operators': {}, 'identity':records[0].get('identity'),
              'qualification': 'CA-calibrated stage surrogate; no physical peak or cross-generation validation'}
    for op in ('quant', 'matmul', 'fa'):
        data = [r for r in records if r['case']['op']==op and r['status']=='ok']
        if len(data) < 3:
            raise ValueError(f'{op}: need 3 successful calibration shapes, have {len(data)}')
        plans = [plan(Case(op, tuple(r['case']['shape']))) for r in data]
        measured = [measured_stages(r) for r in data]
        costs = {}
        for stage in ('read','vector','cube','fix','write'):
            xs = [p['units'][stage] for p in plans]
            ys = [m[stage] / (p['iterations'] * p['waves']) for p,m in zip(plans,measured)]
            costs[stage] = fit_nonnegative_line(xs,ys) if any(xs) else {'intercept_us':0.,'slope_us_per_unit':0.}
        bodies = [schedule(op,p,{k:line(costs[k],u) for k,u in p['units'].items()}) for p in plans]
        refs = [r['measurement']['reference_us'] for r in data]
        correction = fit_nonnegative_line(bodies, refs)
        # Diagnostic instruction-count slopes (not an ISA throughput model).
        opcodes = sorted({k for r in data for c in r['measurement']['cores'] for k in c['dynamic_opcodes']})
        counts = {}
        work = [p['work_per_core'] for p in plans]
        for opcode in opcodes:
            ys = [max(c['dynamic_opcodes'].get(opcode,0) for c in r['measurement']['cores']) for r in data]
            counts[opcode] = fit_nonnegative_line(work,ys)
        shape_ranges = [[min(r['case']['shape'][i] for r in data), max(r['case']['shape'][i] for r in data)] for i in range(len(data[0]['case']['shape']))]
        # Roofline-like empirical baseline: max(compute_work/rate, bytes/rate).
        # Rates derive from calibration makespan, not advertised hardware peaks.
        work_rate = max(p['useful_work']/t for p,t in zip(plans,refs))
        byte_rate = max(p['logical_bytes']/t for p,t in zip(plans,refs))
        result['operators'][op] = {
            'stage_costs': costs, 'total_correction': correction,
            'opcode_count_models': counts, 'shape_ranges': shape_ranges,
            'calibration_ids': [r['case']['id'] for r in data],
            'kernels': sorted({r['measurement']['kernel'] for r in data}),
            'constant_baseline_us': sum(refs)/len(refs),
            'effective_work_per_us': work_rate, 'effective_bytes_per_us': byte_rate,
            'mapping_checks': [{'case':r['case']['id'],'predicted':[p['active_aic'],p['active_aiv']],
                                'observed':[r['measurement']['active_aic'],r['measurement']['active_aiv']]} for p,r in zip(plans,data)],
        }
    return result


class Predictor:
    def __init__(self, model, cache_size=4096):
        self.model = freeze(json.loads(Path(model).read_text()) if isinstance(model,(str,Path)) else model)
        if self.model['soc_version'] != 'Ascend950DT_9582':
            raise ValueError('This version is calibrated for 950DT only')
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def predict(self, case, use_cache=True):
        if use_cache and case in self.cache:
            self.cache.move_to_end(case)
            return self.cache[case]
        p = plan(case)
        model = self.model['operators'][case.op]
        stages = {k: line(model['stage_costs'][k],u) for k,u in p['units'].items()}
        body = schedule(case.op,p,stages)
        estimate = line(model['total_correction'],body)
        if estimate <= 0:
            raise ValueError('Invalid nonpositive prediction')
        inside = all(a<=d<=b for d,(a,b) in zip(case.shape,model['shape_ranges']))
        out = {'op':case.op, 'shape':list(case.shape), 'predicted_us':estimate,
               'reference_scope':'cross-core instruction makespan', 'soc_version':self.model['soc_version'],
               'coverage':'within_calibration_box' if inside else 'extrapolation',
               'kernel_selection_verified':False, 'plan':p, 'stage_services_us':stages,
               'scheduled_body_us':body, 'fixed_overhead_us':model['total_correction']['intercept_us'],
               'body_scale':model['total_correction']['slope_us_per_unit'],
               'constant_baseline_us':model['constant_baseline_us'],
               'empirical_roofline_us':max(p['useful_work']/model['effective_work_per_us'],p['logical_bytes']/model['effective_bytes_per_us']),
               'uncorrected_schedule_us':body,
               'ideal_compute_efficiency':None,
               'efficiency_unavailable_reason':'No independently validated 950DT per-opcode issue/peak model in v0.1'}
        out = freeze(out)
        if use_cache and self.cache_size:
            self.cache[case] = out
            if len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        return out

    def predict_many(self, cases):
        return [self.predict(case) for case in cases]

    def instruction_counts(self, case):
        p = plan(case)
        return {name:max(0.,line(cost,p['work_per_core'])) for name,cost in self.model['operators'][case.op]['opcode_count_models'].items()}
