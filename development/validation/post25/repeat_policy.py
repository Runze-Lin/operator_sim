"""A shadow policy for one ordered rank/device/stream, evaluated before successful-state update."""
from dataclasses import dataclass
from operator_model.post_resource import resource_post
from operator_model.post_token_pipeline import token_pipeline

@dataclass(frozen=True)
class Contract:
    # The stream's scheduler owns these canonical names. Pointers and tensor values are not inputs.
    rank_stream: tuple
    chip: object
    op: str
    implementation: str
    shape: tuple
    dtype: str
    layout: str
    has_h_res: bool

def estimate_repeat(contract, last_success, resident_aiv, prices):
    """Price the built-in BF16 contract. Unknown, ordinary reuse or other preceding op uses v0.24."""
    if (contract.op,contract.implementation,contract.dtype,contract.layout,contract.has_h_res)!=('mhc_post','aclnnMhcPost','bf16','contiguous_ND',True):
        raise ValueError('Route this contract through its own installed backend')
    attrs={} if resident_aiv is None else {'resident_aiv':resident_aiv}
    old,_,a,diag=resource_post(contract.shape,contract.chip,attrs)
    if not (last_success==contract and diag['context']=='reuse' and diag['target_plan']['route']=='post_token'):
        return old,'published v0.24'
    pipe=token_pipeline(contract.shape,contract.chip,diag['target_plan'],'repeat',prices,diag['shared_floor_us'])
    return pipe['us'],'immediate identical builtin Post'
