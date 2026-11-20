"""Structural plans for installed Ascend 950 operator families."""
from operator_model.extensions import norm_addmm
from operator_model.extensions import indexer_mhc
from operator_model.extensions import kda_moe

OPERATORS={
    'rmsnorm':norm_addmm.plan,'rms_norm':norm_addmm.plan,'addmm':norm_addmm.plan,
    'lightning_indexer':indexer_mhc.plan,'lightening_indexer':indexer_mhc.plan,'lightningindexer':indexer_mhc.plan,
    'mhc':indexer_mhc.plan,'mhc_pre':indexer_mhc.plan,'mhc_sinkhorn':indexer_mhc.plan,
    'mhc_pre_sinkhorn':indexer_mhc.plan,'mhc_post':indexer_mhc.plan,
    **{name:kda_moe.plan for name in kda_moe.OPS},
}

def plan(op,shape,**attrs):
    """Return the structural plan. The public estimate API selects the timing model."""
    if op not in OPERATORS:raise ValueError('Unknown extension operator: '+op)
    return OPERATORS[op](op,shape,**attrs)
