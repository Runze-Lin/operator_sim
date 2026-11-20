"""Tile rounding, quantization partition and first-to-last stage scheduling."""
import math


def ceildiv(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceildiv(a, b) * b


def quant_partition(rows, cols, capacity=64):
    """Documented tail-axis block32 2D split, independently expressed in Python.

    Source: ops-nn arch35 DynamicMxQuantTailAxisTiling (see evidence-v1.md).
    These are source-derived hypotheses until checked against the packaged kernel.
    """
    width = 256 if rows * ceildiv(cols, 512) <= capacity * .75 else 512
    columns = ceildiv(cols, width)
    cores = min(capacity, rows * columns)
    divisors = {v for m in range(1, math.ceil(math.sqrt(cores) + 1)) for v in (m, cores // m)}
    options = []
    for rsplit in sorted(divisors):
        csplit = max(1, cores // rsplit)
        if rsplit > rows or csplit > columns:
            continue
        nr, nc = ceildiv(rows, rsplit), ceildiv(columns, csplit)
        hole = nr * nc
        if rsplit * csplit == cores:
            if rows % rsplit == 0 and columns % csplit == 0:
                hole = 0
            elif rows % rsplit == 0:
                hole -= nr * (columns % nc)
            elif columns % csplit == 0:
                hole -= (rows % nr) * nc
            else:
                hole -= (rows % nr) * (columns % nc)
        options.append((hole, csplit, rsplit, nr, nc))
    _, _, _, nr, nc = min(options)
    return {'active_aiv': ceildiv(rows, nr) * ceildiv(columns, nc),
            'rows_per_core': nr, 'cols_per_core': min(cols, nc * width), 'slice_cols': width}


def schedule(op, p, stages):
    """Finite pipeline fill/steady/drain; phase services, not per-instruction replay."""
    read, vector, cube, fix, write = (stages[k] for k in ('read', 'vector', 'cube', 'fix', 'write'))
    if op == 'fa':
        # QK -> softmax -> PV, with shared Cube resource; running softmax/
        # accumulation creates a V+PV recurrence. MTE can overlap next tile.
        fill = read + cube + vector + fix
        ii = max(read, cube + fix, vector + cube / 2)
        return p['waves'] * (fill + (p['iterations'] - 1) * ii + write)
    fill = read + vector + cube + fix + write
    ii = max(read, vector, cube, fix, write)
    return fill + (p['waves'] - 1) * ii


def line(cost, x):
    return cost['intercept_us'] + cost['slope_us_per_unit'] * x
