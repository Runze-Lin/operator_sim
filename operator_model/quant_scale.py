"""Estimate the fan-in of distinct AIV producers to one aligned scale granule."""
from math import gcd


def peak_scale_writers(rows, cols, core_rows, core_cols, granule=32):
    """Count distinct owners by intersecting boundaries, with periodic residues.

    Each scale byte covers 32 adjacent elements. Full-width row blocks are
    contiguous. For column splits the first and last grain of each producer
    interval suffice: interior grains cannot have multiple owners.
    """
    stride = (cols+31)//32
    width = (core_cols+31)//32
    size = rows*stride
    if width>=stride:
        length=core_rows*stride
        if length%granule==0 or size<=length:
            return 1
        # First bytes in equivalent granules have the same block-offset residue.
        return max((min(i+granule,size)-1)//length-i//length+1
                   for i in range(0,min(size,length*granule//gcd(granule,length)),granule))
    if stride%granule==0 and width%granule==0:
        return 1
    # Residues repeat independently along the two grid dimensions. Include
    # starts/ends of row partitions to capture a granule crossing two blocks.
    row_period=granule//gcd(stride,granule)
    row_blocks=min((rows+core_rows-1)//core_rows,row_period)
    column_blocks=min((stride+width-1)//width,granule//gcd(width,granule))
    edge=set()
    for b in range(row_blocks):
        start=b*core_rows
        stop=min(rows,start+core_rows)
        for row in {*range(start,min(stop,start+row_period)),stop-1}:
            for column in (*range(0,column_blocks*width,width),stride-1):
                edge.add((row*stride+column)//granule*granule)
    return max(len({((offset//stride)//core_rows,(offset%stride)//width)
                    for offset in range(i,min(i+granule,size))}) for i in edge)
