import argparse
import json
from .model import Case, Predictor


def main():
    parser = argparse.ArgumentParser(description='Predict without CANN or CA installed')
    parser.add_argument('--model', default='results/v0.1/model.json')
    parser.add_argument('--op', required=True, choices=['quant','matmul','fa'])
    parser.add_argument('--shape', required=True, type=int, nargs='+')
    parser.add_argument('--instruction-counts', action='store_true')
    args = parser.parse_args()
    model=json.loads(open(args.model).read())
    if model['schema_version']==2:
        from .model_v2 import Case as SelectedCase, Predictor as SelectedPredictor
    else:
        SelectedCase,SelectedPredictor=Case,Predictor
    predictor = SelectedPredictor(model)
    case = SelectedCase(args.op, tuple(args.shape))
    output = predictor.predict(case)
    if args.instruction_counts:
        output = {**output, 'instruction_count_diagnostics':predictor.instruction_counts(case)}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
