"""Reference-chip price book with distinct quant, MMAD and Flash Attention schedules."""
from math import log2

from operator_model.profiles import attention_loop
from operator_model.profiles.quant_dispatch import applies, correction
from operator_model.profiles.stage_model import ceildiv, line
from operator_model.profiles.tile_model import body_time, features, plan


class ReferencePricing:
    def __init__(self, parameters):
        self.tiles = {op: parameters[op] for op in ('fa', 'matmul')}
        quant = parameters['quantization']
        self.quant_costs = quant['costs']
        self.quant_opcode_weights = quant['opcode_weights']
        self.control_log2_cap = quant['control_log2_cap']
        self.issue_gap = quant['issue_gap']
        self.attention_costs = parameters['attention_loop']

    def tile_schedule(self, case):
        tile = plan(case)
        parameters = self.tiles[case.op]
        stages = {k: line(parameters['stage_costs'][k], units) for k, units in tile['units'].items()}
        body = body_time(case.op, tile, stages)
        base = sum(a * b for a, b in zip(features(tile, body), parameters['total_weights']))
        offset = parameters['branch_offsets'].get(tile['branch'], 0)
        return {'predicted_us': max(.001, base + offset), 'plan': tile, 'stage_services_us': stages}

    def predict_us(self, case):
        if case.op == 'quant':
            return self.quant_scalar(case)
        if attention_loop.supported(case):
            return attention_loop.schedule(case.shape[1], self.attention_costs)['predicted_us']
        return self.tile_schedule(case)['predicted_us']

    def instruction_counts(self, case):
        tiles = plan(case)
        if case.op == 'quant':
            rows, columns = tiles['rows_per_core'], tiles['cols_per_core']
            height, width = tiles['tile']
            features = [0, 0, 0]
            # Four rectangles cover the full tile grid, its two edges and the corner.
            for h, ny in ((height, rows // height), (rows % height, 1)):
                for w, nx in ((width, columns // width), (columns % width, 1)):
                    if h and w and nx and ny:
                        groups = h * ceildiv(w, 32)
                        for i, count in enumerate((1, ceildiv(groups, 8), ceildiv(groups, 128))):
                            features[i] += nx * ny * count
            value = sum(a * b for a, b in zip(features, self.quant_opcode_weights))
            family = 'RV_* including load/store/control'
        elif case.op == 'fa':
            value = 2 * ceildiv(case.shape[1], 128) * tiles['waves'] if tiles['q_tile_capacity'] == 64 else None
            family = 'MMAD'
        else:
            value = sum(a * b for a, b in zip(tiles['opcode_features'], self.tiles['matmul']['opcode_weights']))
            family = 'MMAD'
        return {'family': family, 'estimated_max_core_count': value}

    def quant_scalar(self, case):
        """Walk a bounded number of tiles; long runs use max-plus exponentiation."""
        tile = plan(case)
        costs = self.quant_costs
        log_cores = log2(1 + tile['active_aiv'])
        prefix = line(costs['prefix'], log_cores)
        floor = line(costs['control_floor'], min(self.control_log2_cap, log_cores))
        if tile['iterations'] == 1:
            total = (prefix + line(costs['first_read'], tile['units']['read'])
                     + line(costs['vector'], tile['units']['vector'])
                     + costs['write_drain_us'] + costs['suffix_us'])
            return total + correction(tile, self.issue_gap) if applies(tile) else max(total, floor)

        rows, cols = tile['rows_per_core'], tile['cols_per_core']
        height, width = tile['tile']
        last_read = last_vector = last_write = older_vector = older_write = 0.
        index = 0
        for row in range(0, rows, height):
            for col in range(0, cols, width):
                h, w = min(height, rows - row), min(width, cols - col)
                read_kib = 2 * h * w / 1024
                groups = h * ceildiv(w, 32)
                latency = line(costs['first_read'], read_kib)
                ready = prefix + latency if index == 0 else last_read + line(costs['read_interval'], read_kib)
                if index >= 2:
                    ready = max(ready, older_vector + latency)
                begin = max(ready, last_vector if index else 0, older_write if index >= 2 else 0)
                end = begin + line(costs['vector'], ceildiv(groups, 8) + ceildiv(groups, 128))
                written = max(end, last_write if index else 0) + costs['write_drain_us']
                older_vector, older_write = last_vector, last_write
                last_read, last_vector, last_write = ready, end, written
                index += 1
        return max(last_write + costs['suffix_us'], floor)
