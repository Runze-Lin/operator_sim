import argparse
import json
from pathlib import Path
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
    parser.add_argument('--chip', help='Built-in chip name or path to a chip JSON config')
    parser.add_argument('--dtype', help='Primary input dtype, e.g. bf16 or fp8_e4m3fn')
    parser.add_argument('--timing-mode', choices=('auto','resource','calibrated'), default='auto')
    args = parser.parse_args()
    attrs=json.loads(args.attrs_json)
    if args.model is None and not args.local_proxy and args.hardware_profile is None:
        from .backend import _evaluate
        from .chips import REFERENCE_NAME
        estimated, raw = _evaluate(args.op, args.shape, attrs, REFERENCE_NAME, args.chip, args.dtype, args.timing_mode)
        # Keep historical DT diagnostics for tools that inspect stage-specific fields.
        # Reference timings must never appear as target-chip phase measurements.
        output = dict(raw) if estimated.projection is None and raw else {}
        output.update(op=args.op, shape=args.shape, soc_version=estimated.chip_name,
                      predicted_us=estimated.latency_us, estimate_status=estimated.status,
                      reason=estimated.reason, work=estimated.work, rates=estimated.rates,
                      active_aic=estimated.active_aic, active_aiv=estimated.active_aiv,
                      projection=estimated.projection)
        output.setdefault('status', estimated.status)
        if args.instruction_counts and estimated.projection is not None:
            parser.error('--instruction-counts is available for the 950DT reference only')
        if args.instruction_counts and output['predicted_us'] is not None and args.op in ('quant','matmul','fa'):
            from .predictor import _legacy, Case as LegacyCase
            output['instruction_count_diagnostics'] = _legacy().instruction_counts(LegacyCase(args.op,tuple(args.shape)))
        print(json.dumps(output,indent=2))
        return
    if args.chip is not None or args.dtype is not None or args.timing_mode != 'auto':
        parser.error('--chip, --dtype and --timing-mode require the current system backend')
    resource_root = Path(__file__).resolve().parents[1]
    args.model = args.model or resource_root / 'results/v0.7/model.json'
    args.hardware_profile = args.hardware_profile or resource_root / 'configs/hardware-950dt-v7.json'
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
