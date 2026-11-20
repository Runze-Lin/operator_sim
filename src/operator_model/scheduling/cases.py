"""Validate inputs for the reference-device timing and instruction profiles."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    op: str
    shape: tuple

    def __post_init__(self):
        shape=tuple(self.shape)
        if self.op=='fa' and len(shape)==3:shape=shape+(1,1)
        object.__setattr__(self,'shape',shape)
        if self.op not in ('quant','matmul','fa') or len(shape)!={'quant':2,'matmul':3,'fa':5}[self.op]:
            raise ValueError('Expected quant(R,C), matmul(M,N,K), fa(Q,KV,D[,B,H])')
        if any(type(x) is not int or not 1<=x<=1048576 for x in shape):
            raise ValueError('Dimensions must be integers in [1,1048576]')
        if self.op=='quant' and (shape[1]%64 or shape[1]>65536):
            raise ValueError('Quant cols must be a multiple of 64, at most 65536')
        if self.op=='matmul' and max(shape)>4096:raise ValueError('Matmul domain remains <=4096')
        if self.op=='fa' and (shape[2] not in (64,128) or shape[3]>256 or shape[4]>32 or shape[0]>2048):
            raise ValueError('FA proxy: D64/128, B<=256, H<=32, Q<=2048, noncausal SBH')
