import argparse
import json
from . import OPERATORS,plan

def main():
    parser=argparse.ArgumentParser(description='Structural plans only; uncalibrated latency is null')
    parser.add_argument('--list',action='store_true');parser.add_argument('--op',choices=sorted(OPERATORS))
    shape=parser.add_mutually_exclusive_group();shape.add_argument('--shape',type=int,nargs='+');shape.add_argument('--shape-json')
    parser.add_argument('--attrs-json',default='{}');args=parser.parse_args()
    if args.list:print(json.dumps(sorted(OPERATORS),indent=2));return
    if args.op is None or (args.shape is None and args.shape_json is None):parser.error('--op and a shape are required')
    try:
        attrs=json.loads(args.attrs_json)
        if not isinstance(attrs,dict):raise ValueError('attrs must be a JSON object')
        dims=args.shape if args.shape is not None else json.loads(args.shape_json)
        result=plan(args.op,dims,**attrs)
    except (ValueError,TypeError) as exc:parser.error(str(exc))
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
