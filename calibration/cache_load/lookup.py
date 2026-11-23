"""Query qualified cache-load CA observations, including routes without a priced model."""
import argparse
import gzip
import json
from pathlib import Path

from calibration.__main__ import digest, encoded, ingest, require, stage_window, unseal

HERE=Path(__file__).resolve().parent
ALIASES={'950DT':'Ascend950DT_9582','950PR':'Ascend950PR_957b'}


def observations(folder=HERE, dtype=None):
    fp8_rows=(json.loads(gzip.decompress((folder/'fp8-smoke.json.gz').read_bytes()))
              if dtype in (None, 'fp8_e4m3fn') else ())
    for row in fp8_rows:
        stage,=row['stages']
        require(row['api']=='aclnnAscendAntiQuant' and row['chip'] in ALIASES.values()
                and row['shape']==[1,128,512] and row['scale_shape']==[128,1]
                and row['x_dtype']=='fp8_e4m3fn' and row['scale_dtype']=='fp32' and row['output_dtype']=='bf16'
                and row['sqrt_mode'] is False and row['offset'] is None and row['predicted_us'] is None,
                'Unexpected rowwise reference contract')
        require(row['ca_us']==stage_window(stage) and row['correctness']=={'Y':{'checked':65536,'bad':0}}
                and stage['trace_sha256']==digest(encoded(stage['physical_shard_sha256']))
                and set(stage['cores'])==set(stage['physical_shard_sha256']),
                'FP8 complete window, trace identity or numerical evidence differs')
        yield dict(dtype='fp8_e4m3fn',implementation=row['api'],chip=row['chip'],shape=row['shape'],
                   data_role='CA observation only',ca_us=row['ca_us'],predicted_us=None,
                   extrapolated_970dt_us=None,active_aiv=len(stage['cores']),kernel=stage['kernel'],
                   image_sha256=stage['kernel_binary_sha256'],wall_seconds=row['wall_seconds'])
    if dtype=='fp8_e4m3fn':
        return
    for name in ('fp4-development.json.gz','fp4-independent.json.gz'):
        archive=folder/name
        bundle=json.loads(gzip.decompress(archive.read_bytes()))
        frozen=unseal(bundle['freeze'],'freeze')
        cases={c['id']:c for c in frozen['cases']}
        for item in bundle['observations']:
            rec=item['receipt']
            measure=ingest(frozen,rec)
            expected='prospective' if name=='fp4-independent.json.gz' else 'replay'
            require(item['measurement']==measure and measure['data_role']==expected,
                    'FP4 measurement or acquisition role differs from its bound receipt')
            c=cases[rec['case']['id']]
            stage,=rec['stages']
            yield dict(dtype=c['dtype'],implementation=c['api'],chip=c['chip'],shape=c['shape'],
                       data_role='independent' if measure['data_role']=='prospective' else 'replay',
                       ca_us=measure['ca_us'],predicted_us=c['predicted_us'],
                       extrapolated_970dt_us=c['extrapolated_970dt_us'],active_aiv=len(stage['cores']),
                       kernel=stage['kernel'],image_sha256=stage['kernel_binary_sha256'],
                       wall_seconds=item['wall_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chip',help='950DT, 950PR or the full SKU; no chip filter by default')
    p.add_argument('--shape',type=int,nargs='+',help='Exact logical output dimensions')
    p.add_argument('--dtype',choices=['mxfp4','fp8_e4m3fn'])
    p.add_argument('--folder',type=Path,default=HERE)
    a=p.parse_args()
    chip=ALIASES.get(a.chip,a.chip)
    rows=[r for r in observations(a.folder,a.dtype) if (not chip or r['chip']==chip) and
          (not a.shape or r['shape']==a.shape)]
    print(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False))
    if not rows:
        p.exit(1,'No matching qualified CA observation.\n')


if __name__=='__main__':
    main()
