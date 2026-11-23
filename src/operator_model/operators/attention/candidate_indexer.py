"""Work accounting for the rank-local DeepSeek candidate indexer capture."""


def describe(shape, attrs):
    if len(shape) != 7 or any(type(d) is not int or d < 1 for d in shape):
        raise ValueError('Expected positive B,Q,H,D,L,C,K for candidate_filtered_indexer')
    b, q, h, d, length, slots, top_k = shape
    rows = b * q
    policy = attrs.get('score_policy', 'full_scan')
    if set(attrs) - {'score_policy', 'block_size', 'scored_keys_per_query'}:
        raise ValueError('Unknown candidate_filtered_indexer attribute')

    # The integer input indexes blocks. Its width is not a count of unique keys.
    if policy == 'full_scan':
        if 'block_size' in attrs or 'scored_keys_per_query' in attrs:
            raise ValueError('block_size/scored_keys_per_query apply only to candidate_first')
        pairs = upper_pairs = rows * length
        key_read = 2 * b * length * d
        traffic_scope = 'Full key tensor read once per request, as in the captured system reference.'
    elif policy == 'candidate_first':
        span = attrs.get('block_size')
        if type(span) is not int or span < 1:
            raise ValueError('candidate_first requires positive integer block_size')
        limit = min(length, slots * span)
        upper_pairs = rows * limit
        counts = attrs.get('scored_keys_per_query')
        if counts is None:
            pairs = None
        elif type(counts) is int and 0 <= counts <= limit:
            pairs = rows * counts
        elif (isinstance(counts, (list, tuple)) and len(counts) == rows
              and all(type(x) is int and 0 <= x <= limit for x in counts)):
            pairs = sum(counts)
        else:
            raise ValueError('Expected a valid scored-key count per query within the block and table bounds')
        key_read = None if pairs is None else 2 * pairs * d
        traffic_scope = 'Logical key gathers per query; inter-query reuse and physical traffic are unknown.'
    else:
        raise ValueError('score_policy must be full_scan or candidate_first')

    fixed_read = 2 * rows * h * d + 2 * rows * h + 8 * rows + 4 * rows * slots
    output = 4 * rows * top_k
    work = {
        'dtype': 'bf16', 'cube_ops': None, 'vec_ops': None,
        'cube_ops_by_dtype': None, 'vec_ops_by_dtype': None,
        'score_dot_ops': None if pairs is None else 2 * h * d * pairs,
        'score_dot_ops_upper_bound': 2 * h * d * upper_pairs,
        'candidate_slots': rows * slots,
        'logical_read_bytes': None if key_read is None else fixed_read + key_read,
        'logical_write_bytes': output, 'physical_hbm_bytes': None,
        'key_table_capacity_bytes': 2 * b * length * d,
        'scope': traffic_scope,
        'notes': ['FMA counts as two. Device-unit assignment, head weighting, mask and top-k instruction costs are unknown.'],
    }
    return work, {
        'score_policy': policy, 'query_rows': rows, 'scored_query_key_pairs': pairs,
        'scored_query_key_pairs_upper_bound': upper_pairs,
        'input_shapes': [[rows, h, d], [b, length, d], [rows, h], [rows], [rows, slots]],
        'output_shape': [rows, top_k], 'timing_evidence': 'system shape capture only',
    }
