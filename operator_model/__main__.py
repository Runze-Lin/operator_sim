import argparse
import json
from .model import Case, Predictor


def main():
    parser = argparse.ArgumentParser(description='Predict without CANN or CA installed')
    parser.add_argument('--model', default='results/v0.6/model.json')
    parser.add_argument('--op', required=True, choices=['quant','matmul','fa'])
    parser.add_argument('--shape', required=True, type=int, nargs='+')
    parser.add_argument('--instruction-counts', action='store_true')
    args = parser.parse_args()
    model=json.loads(open(args.model).read())
    if model['schema_version']==6:
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
    output = predictor.predict(case)
    if args.instruction_counts:
        output = {**output, 'instruction_count_diagnostics':predictor.instruction_counts(case)}
    if args.op=='quant':
        output={**output,'selected_opcode_count_scope':'legacy msopprof RV log events, not unique helper ExecutedInstructions; see docs/error-analysis.md'}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
