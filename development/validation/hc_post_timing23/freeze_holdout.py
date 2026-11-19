"""Register dense BF16 inputs with safe 32-byte row strides and model checksums."""
from pathlib import Path
import json
import series
from model import predict

series.HOLDOUT=[(1,512),(2,1280),(4,4080),(8,4096),(7,4112),
                (33,7168),(2,8192),(1,16384),(57,192),(129,128)]


def main():
    primary=series.OUT/'holdout-register.json'
    if primary.exists():raise FileExistsError(primary)
    # Shape + attributes must differ from every development window on either end.
    old=series.read(series.ORIGINAL_REG)
    seen={(c['tokens'],c['hidden']) for c in old['cases']}|{(t,d) for _,t,d in series.DEVELOPMENT}
    assert len(set(series.HOLDOUT))==10 and not seen.intersection(series.HOLDOUT)
    for t,d in series.HOLDOUT:
        for chip in (series.DT,series.PR,'Ascend970DT'):
            r=predict(t,d,chip);assert r['active_aiv']>0 and r['device_us']>0
        for chip in (series.DT,series.PR):
            key=f'holdout-{chip}-t{t}-d{d}'
            assert not (series.RAW/key).exists() and not (series.OUT/(key+'.json')).exists()
    files=[Path(__file__),series.HERE/'calibrate.py',series.HERE/'extract.py',
           *(series.ROOT/'configs/chips'/f'{k}.json' for k in (series.DT,series.PR,'Ascend970DT'))]
    extension=dict(utc=series.utc(),files=series.hashes(files),shapes=series.HOLDOUT,
                   source_note='Source uses aligned D for residual/output GM row stride; restrict dense BF16 to D multiple of 16 (32-byte data block). The near-edge rows include masked VF repeats.',
                   measurement_order='shape order, each sequentially DT then PR',
                   passing='same installed extension ELF, complete all-core trace, complete exact BF16 golden, full earliest-to-latest device window')
    with (series.OUT/'holdout-inputs.json').open('x') as f:json.dump(extension,f,indent=2);f.write('\n')
    series.register('holdout')


if __name__=='__main__':main()
