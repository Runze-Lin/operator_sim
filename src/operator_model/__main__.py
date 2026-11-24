"""Command-line interface for a single operator estimate."""
import argparse
import json

from operator_model.backend import _evaluate, _memory_window
from operator_model.chips import REFERENCE_NAME, load_chip


def main():
    parser = argparse.ArgumentParser(description='Predict Ascend operator work and latency without CANN or CA')
    parser.add_argument('--op', required=True, help='Operator name, e.g. quant, matmul or fa')
    parser.add_argument('--shape', required=True, type=int, nargs='+')
    parser.add_argument('--attrs-json', default='{}', help='Operator attributes as JSON')
    parser.add_argument('--chip', help='Built-in chip name or path to a chip JSON config')
    parser.add_argument('--dtype', help='Primary input dtype, e.g. bf16 or fp8_e4m3fn')
    parser.add_argument('--timing-mode', choices=('auto', 'resource', 'calibrated'), default='auto')
    parser.add_argument('--memory-policy', choices=('ca_window', 'cold_io'), default='ca_window')
    parser.add_argument('--instruction-counts', action='store_true', help='Include reference-chip instruction diagnostics')
    args = parser.parse_args()

    attrs = json.loads(args.attrs_json)
    estimated, raw = _evaluate(args.op, args.shape, attrs, REFERENCE_NAME, args.chip, args.dtype, args.timing_mode)
    reference_diagnostics = estimated.projection is None
    if args.memory_policy == 'cold_io' and estimated.latency_us is not None:
        estimated = _memory_window(estimated, load_chip(args.chip or REFERENCE_NAME), 'cold_io')
    output = dict(raw) if reference_diagnostics and raw else {}
    output.update(op=args.op, shape=args.shape, soc_version=estimated.chip_name,
                  predicted_us=estimated.latency_us, estimate_status=estimated.status,
                  kernel_count=estimated.kernel_count, dispatch_overhead_us=estimated.dispatch_overhead_us,
                  scheduled_us=estimated.scheduled_us, reason=estimated.reason, work=estimated.work, rates=estimated.rates,
                  active_aic=estimated.active_aic, active_aiv=estimated.active_aiv,
                  projection=estimated.projection)
    output.setdefault('status', estimated.status)
    if args.instruction_counts and not reference_diagnostics:
        parser.error('--instruction-counts is available for the 950DT reference only')
    if args.instruction_counts and output['predicted_us'] is not None and args.op in ('quant', 'matmul', 'fa'):
        from operator_model.scheduling.reference import reference_prices
        from operator_model.scheduling.cases import Case
        output['instruction_count_diagnostics'] = reference_prices().instruction_counts(Case(args.op, tuple(args.shape)))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
