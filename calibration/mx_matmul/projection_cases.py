"""Rank-local geometries for the paired dense MX projection campaign."""
import json
from pathlib import Path
from calibration import __main__ as protocol


# Decode extent is tokens verified per request. Prefill extent is the batched
# chunk budget for the whole replica of eight 32-Ki-token requests.
PROJECTIONS = (
    ('DeepSeek-V4.1-Flash', 'dsa.wkv', 'decode', 1, 1),
    ('Kimi-K3', 'mla.q_b_proj', 'decode', 1, 9),
    ('DeepSeek-V4-Pro', 'dsa.wq_b', 'decode', 1, 11),
    ('DeepSeek-V4-Pro', 'moe.shared_down', 'decode', 16, 1),
    ('Kimi-K3', 'mla.kv_b_proj', 'prefill', 8, 128),
    ('DeepSeek-V4-Pro', 'indexer.compressor.kv', 'prefill', 8, 256),
    ('DeepSeek-V4-Pro', 'indexer.weights_proj', 'prefill', 8, 512),
    ('DeepSeek-V4.1-Flash', 'indexer.wk', 'prefill', 8, 1024),
    ('DeepSeek-V4.1-Flash', 'indexer.weights_proj', 'prefill', 8, 2048),
    ('DeepSeek-V4.1-Flash', 'indexer.wk', 'prefill', 8, 4096),
)
SKUS = ('Ascend950DT_9582', 'Ascend950PR_957b')


def manifests():
    from examples.workloads.prefill import MODELS_DIR, Scenario, contracts

    sources = {s['repo'].split('/')[-1]: s for s in protocol.read(MODELS_DIR/'sources.json')}
    cases, shapes = [], []
    for index, (model, op, phase, batch, extent) in enumerate(PROJECTIONS, 1):
        # Linear projection width and row geometry are independent of the
        # cache history. Nonlinear attention itself cannot use this shortcut.
        m = batch * extent if phase == 'decode' else extent
        scenario = (f'decode B{batch} Q{extent}' if phase == 'decode' else
                    f'prefill B{batch} chunk{extent}')
        dispatch = Scenario(model, 32768, batch=batch, chunk=m)
        matching = [row['tensors'] for row in contracts(dispatch)['contracts'] if row['op'] == op]
        shapes_for_op = {(t['x'][0], *t['weight_logical']) for t in matching}
        if len(shapes_for_op) != 1:
            raise ValueError(f'Ambiguous projection geometry: {model} {op}')
        (rows, n, k), = shapes_for_op
        if rows != m:
            raise ValueError(f'Projection does not consume the requested token rows: {model} {op}')
        geometry = [m, n, k]
        tag = f'p{index:02d}'
        shapes.append(dict(id=tag, model=model, projection=op, shape=geometry, scenario=scenario,
                           tp=8, source=sources[model], evidence='rank-local projection geometry; both MX precision policies tested as scenarios'))
        for route, dtype, attrs in [('native','mxfp4',{}), ('mixed','fp8_e4m3fn',{'weight_dtype':'mxfp4'})]:
            for chip, suffix in zip(SKUS, ('dt','pr')):
                cases.append(dict(id=f'{route}-{tag}-{suffix}',op='matmul',contract='matmul_mx_'+route,
                                  chip=chip, shape=geometry, attrs=attrs, dtype=dtype, prefix=None,
                                  api='aclnnQuantMatmulV5',outputs={'Y':m*n}))
    if len({tuple(s['shape']) for s in shapes})!=len(PROJECTIONS):
        raise ValueError('Projection geometries must be distinct')
    return dict(schema=1,cases=cases),shapes


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output_dir',type=Path)
    directory=cli.parse_args().output_dir
    directory.mkdir(parents=True,exist_ok=True)
    cases, shapes = manifests()
    for name,data in [('cases',cases),('geometries',shapes)]:
        with (directory/f'{name}.json').open('x') as dest:
            json.dump(data,dest,indent=2)
            dest.write('\n')
