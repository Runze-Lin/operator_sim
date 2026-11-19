"""Repeat the sole qualified Cast window whose output DMA bytes were incomplete."""
import json
import os
import sys
from cast_validate import OUT,RAW,BINARY,PROFILE,capture,digest,now,read,stages,write
from cast_io import inspect


def main(collect):
    base=read(OUT/'cast-holdout-freeze.json'); path=OUT/'cast-dma-recheck-freeze.json'
    if not collect:
        suspect=[r for r in read(OUT/'cast-dma-coverage.json') if r['write_bytes_observed']!=r['logical_write_bytes']]
        assert len(suspect)==1 and not path.exists()
        r=suspect[0]
        job=next(j for j in base['jobs'] if j['shape']==r['shape'])
        write(path,dict(registered=now(),job=job,sku=r['sku'],earlier=r,
                        reason='Repeat all and only full-window holdouts with mismatched logical vs descriptor output bytes; preserve primary statistics separately.',
                        binary_sha256=base['binary_sha256'],module_sha256=base['module_sha256'],omp_wait=base['omp_wait']))
        return
    reg=read(path);job=reg['job'];sku=reg['sku']
    assert digest(BINARY)==reg['binary_sha256'] and digest(PROFILE)==reg['module_sha256']
    os.environ.update(OMP_WAIT_POLICY=reg['omp_wait'],GOMP_SPINCOUNT='0')
    folder=RAW/(job['id']+'_'+sku+'-dma-check')
    cost=read(folder/'cost.json') if (folder/'cost.json').exists() else capture(job,sku,folder)
    row=dict(shape=job['shape'],sku=sku,cost=cost)
    try:
        ks,checks=stages(job,sku,folder)
        row.update(qualified=True,kernel=ks[0],correctness=checks)
        row['dma']=inspect(row)
        row['complete_dma']=all(row['dma'][k+'_bytes_observed']==row['dma']['logical_'+k+'_bytes'] for k in ('read','write'))
    except (ValueError,AssertionError,IndexError) as e:
        row.update(qualified=False,complete_dma=False,reason=str(e))
    write(OUT/'cast-dma-recheck.json',row)
    print(row['qualified'],row['complete_dma'])


if __name__=='__main__':
    if sys.argv[1:] not in (['collect'],['register']):raise ValueError('Expected register or collect')
    main(sys.argv[1]=='collect')
