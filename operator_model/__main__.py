import argparse
import json
from .model import Case, Predictor


def main():
    parser = argparse.ArgumentParser(description='Predict without CANN or CA installed')
    parser.add_argument('--model', default=None)
    parser.add_argument('--op', required=True, help='quant, matmul, fa, or an extension operator')
    parser.add_argument('--shape', required=True, type=int, nargs='+')
    parser.add_argument('--instruction-counts', action='store_true')
    parser.add_argument('--attrs-json',default='{}',help='Operator attributes as JSON')
    parser.add_argument('--local-proxy',action='store_true',help='Opt in to a local diagnostic baseline, not the general mechanism model')
    parser.add_argument('--hardware-profile',default=None)
    args = parser.parse_args()
    attrs=json.loads(args.attrs_json)
    if args.model is None and not args.local_proxy and args.hardware_profile is None:
        from .predictor import predict
        output = predict(args.op,args.shape,attrs)
        if args.instruction_counts and output['predicted_us'] is not None and args.op in ('quant','matmul','fa'):
            from .predictor import _legacy, Case as LegacyCase
            output['instruction_count_diagnostics'] = _legacy().instruction_counts(LegacyCase(args.op,tuple(args.shape)))
        print(json.dumps(output,indent=2))
        return
    args.model = args.model or 'results/v0.7/model.json'
    args.hardware_profile = args.hardware_profile or 'configs/hardware-950dt-v7.json'
    if args.op in ('quant','matmul','fa') and attrs:
        parser.error('Legacy quant/matmul/fa models have fixed attributes; attrs-json is for extension contracts')
    if args.op not in ('quant','matmul','fa') and not args.local_proxy:
        from .mechanism import predict_case
        profile=json.loads(open(args.hardware_profile).read())
        print(json.dumps(predict_case(args.op,args.shape,attrs,profile),indent=2))
        return
    model=json.loads(open(args.model).read())
    if model['schema_version']==7:
        from .model_v7 import Case as SelectedCase, Predictor as SelectedPredictor
    elif model['schema_version']==6:
        from .model_v6 import Case as SelectedCase
        from .fast_query import Predictor as SelectedPredictor
    elif model['schema_version']==5:
        from .model_v5 import Case as SelectedCase, Predictor as SelectedPredictor
    elif model['schema_version']==4:
        from .model_v4 import Case as SelectedCase, Predictor as SelectedPredictor
    elif model['schema_version']==3:
        from .model_v3 import Case as SelectedCase, Predictor as SelectedPredictor
    elif model['schema_version']==2:
        from .model_v2 import Case as SelectedCase, Predictor as SelectedPredictor
    else:
        SelectedCase,SelectedPredictor=Case,Predictor
    predictor = SelectedPredictor(model)
    case = SelectedCase(args.op, tuple(args.shape))
    output = predictor.predict(case,attrs) if model['schema_version']==7 else predictor.predict(case)
    if args.instruction_counts:
        output = {**output, 'instruction_count_diagnostics':predictor.instruction_counts(case)}
    if args.op=='quant':
        output={**output,'selected_opcode_count_scope':'legacy msopprof RV log events, not unique helper ExecutedInstructions; see docs/error-analysis.md'}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
