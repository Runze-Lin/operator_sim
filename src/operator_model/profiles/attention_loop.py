"""Flash Attention fill, loop issue interval and drain for long sequences."""
from operator_model.profiles.stage_model import ceildiv


def supported(case):
    return case.op == 'fa' and case.shape[0] == 9 and case.shape[1] >= 4096 and case.shape[2:] == (64, 1, 1)


def schedule(length, costs):
    iterations = ceildiv(length, 256)
    # Initial K/V read streams are staggered. Their windows may overlap.
    read_fill = max(costs['first_k_latency_us'], costs['first_v_start_lag_us'] + costs['first_v_latency_us'])
    # Initiation intervals already include steady scheduling/backpressure.
    interval = max(costs['k_initiation_interval_us'], costs['v_initiation_interval_us'],
                   costs['cube_service_per_tile_us'], costs['vector_service_per_tile_us'],
                   costs['mte1_service_per_tile_us'], costs['fix_service_per_tile_us'])
    total = costs['prefix_us'] + read_fill + costs['compute_fill_us'] + (iterations-1)*interval + costs['drain_us']
    return {'predicted_us': total, 'iterations': iterations, 'prefix_us': costs['prefix_us'],
            'read_fill_us': read_fill, 'compute_fill_us': costs['compute_fill_us'],
            'steady_interval_us': interval, 'steady_total_us': (iterations-1)*interval,
            'drain_us': costs['drain_us'], 'tail_tokens': length % 256,
            'tail_policy': 'charge a full 256-token tile; independently check tail dispatch'}
