"""Reproduce working-set geometry, resource budgets and predicted device time."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from operator_model.chips import load_chip
from operator_model.system_routes.cube import _schedule

DEV = ((9,4,20480,'FP32',False),(9,384,5120,'FP32',False),
       (8,1,7680,'FP32',True),(9,128,512,'BF16',False))
HOLDOUT = ((9,4,28672,'FP32',False),(9,384,7168,'FP32',False),
           (128,1,5376,'FP32',True),(144,128,512,'BF16',False))


def predict(cases):
    result = {}
    for sku in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
        chip = load_chip(sku)
        rows = []
        for m,n,k,dt,bias in cases:
            width = 4 if dt=='FP32' else 2
            mode = 'router16' if dt=='BF16_FP32' else ('dense32' if width==4 else 'dense16')
            output_width = 4 if mode=='router16' else width
            args = ((m,),n,k,mode,(m*k+n*k+n*int(bias))*width,m*n*output_width,
                    {'bias_add': m*n} if bias else {},chip,bias)
            start = perf_counter_ns()
            us,schedule = _schedule(*args)
            query_us = (perf_counter_ns()-start)/1000
            rows.append(dict(shape_MNK=(m,n,k),dtype=dt,api='aclnnAddmm' if bias else 'aclnnMm',
                             predicted_us=us,model_query_us=query_us,schedule=schedule))
        result[sku] = rows
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--router',action='store_true',help='Price the BF16-input FP32-output native router contract')
    args=parser.parse_args()
    src=Path(__file__).resolve().parents[2]/'src/operator_model/system_routes/cube.py'
    data=dict(model_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    if args.router:
        for role,m in (('development',9),('independent',144)):
            data[role]={sku:rows[0] for sku,rows in predict(((m,896,7168,'BF16_FP32',False),)).items()}
    else:
        data.update(development=predict(DEV),independent=predict(HOLDOUT))
    args.output.write_text(json.dumps(data,indent=2)+'\n')
