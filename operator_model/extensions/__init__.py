"""Uncalibrated structural plans for additional installed 950 operator families."""
from . import norm_addmm, indexer_mhc, kda_moe

OPERATORS={
    'rmsnorm':norm_addmm.plan,'rms_norm':norm_addmm.plan,'addmm':norm_addmm.plan,
    'lightning_indexer':indexer_mhc.plan,'lightening_indexer':indexer_mhc.plan,'lightningindexer':indexer_mhc.plan,
    'mhc':indexer_mhc.plan,'mhc_pre':indexer_mhc.plan,'mhc_sinkhorn':indexer_mhc.plan,
    'mhc_pre_sinkhorn':indexer_mhc.plan,'mhc_post':indexer_mhc.plan,
    **{name:kda_moe.plan for name in kda_moe.OPS},
}

def plan(op,shape,**attrs):
    """Return work/route facts; no extension has a calibrated latency yet."""
    if op not in OPERATORS:raise ValueError('Unknown extension operator: '+op)
    return OPERATORS[op](op,shape,**attrs)
