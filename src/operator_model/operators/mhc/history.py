"""Conditional timing scenarios for the built-in four-stream BF16 aclnnMhcPost."""
from operator_model.resources import load_parameters
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, fsum, isclose, isfinite, log2
from types import MappingProxyType

from operator_model.backend import _call_attrs
from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.operators.mhc.repeat import _prices as token_prices, channel_tiles
from operator_model.operators.mhc.resources import resource_post
from operator_model.operators.mhc.token_pipeline import token_pipeline
from operator_model.operators.rmsnorm.resources import READ_PER_KIB


@lru_cache(maxsize=1)
def _prices():
    return load_parameters('post_history')


@dataclass(frozen=True)
class PostScenario:
    """One conditional resource estimate, measured on other calls or extrapolated."""
    latency_us: float
    stages_us: Mapping
    model: str
    evidence: str


@dataclass(frozen=True)
class PostHistory:
    """Mutually exclusive prior states for one device, ordered stream and Post shape.

    Times are device microseconds. No distribution is implied. All scenarios are
    estimates; even the entire predicted range may miss an individual CA result.
    """
    shape: tuple
    chip_name: str
    active_aiv: int
    route: str
    scenarios: Mapping
    target_extrapolated: bool
    _chip: object=field(repr=False,compare=False)

    @property
    def range_us(self):
        times=[s.latency_us for s in self.scenarios.values()]
        return min(times),max(times)

    def expected_us(self, probabilities: Mapping[str,float]) -> float:
        """Expected device time for explicit mutually exclusive state probabilities.

        Missing states have zero mass. Supply measured serving frequencies or
        a stated scenario; a set of unspecified weights has no expectation.
        """
        if not isinstance(probabilities,Mapping) or not probabilities:
            raise ValueError('A nonempty mapping of known Post states to probabilities is required')
        pairs=[]
        for state,p in probabilities.items():
            if state not in self.scenarios:
                raise ValueError(f'Unknown or infeasible Post state: {state!r}')
            if not isinstance(p,(int,float)) or isinstance(p,bool) or not isfinite(p) or not 0<=p<=1:
                raise ValueError('Each state probability must be finite and between zero and one')
            pairs.append((self.scenarios[state].latency_us,p))
        if not isclose(fsum(p for _,p in pairs),1.,rel_tol=0.,abs_tol=1e-9):
            raise ValueError('Post state probabilities must sum to one')
        return fsum(t*p for t,p in pairs)

    def for_predecessor(self,resident_aiv:int,previous_shape=None) -> PostScenario:
        """Price a fully known family prefix and immediately preceding built-in Post.

        resident_aiv counts the resident same-family high-water mark on this
        rank and stream. Use 0/None for a known empty stream. An unknown
        predecessor or another kernel family must not be relabeled as a repeat.
        """
        if type(resident_aiv) is not int or not 0<=resident_aiv<=self._chip.vector_cores:
            raise ValueError('Invalid same-family resident_aiv count')
        if previous_shape is None:
            if resident_aiv==0: return self.scenarios['first']
            raise ValueError('Previous built-in Post shape is required for a nonempty prefix')
        previous=tuple(previous_shape)
        _,_,prev_active,prev=resource_post(previous,self._chip,{})
        if resident_aiv<prev_active:
            raise ValueError('resident_aiv does not cover the preceding successful Post')
        if previous==self.shape: return self.scenarios['repeat']
        state='reuse' if resident_aiv>=self.active_aiv else 'expand'
        if self.route=='post_token':
            state+='_after_'+prev['target_plan']['route'].removeprefix('post_')
        return self.scenarios[state]


def _scenario(stages,method,evidence):
    return PostScenario(fsum(stages.values()),MappingProxyType(dict(stages)),method,evidence)


def _channel_repeat(plan,chip,shared_floor,vf=None):
    """Two read descriptors at distinct issue times on the verified channel-split path."""
    p=_prices()['channel_repeat'];ref_clock=1.65/chip.frequency_ghz
    phase=p['phase'];port=p['port'];c_per_us=1000*chip.frequency_ghz
    d=plan['fragment_width']
    groups=ceil(d/64)
    vec=(27/64)/(chip.vector_tops['32']/chip.vector_cores)
    stages={
        'arrival':ref_clock*(phase['arrival']['base_us']+phase['arrival']['slopes'][0]*log2(plan['active_aiv'])/6),
        'control':ref_clock*phase['control']['base_us'],
        'body':ref_clock*phase['body']['base_us']+phase['body']['slopes'][0]*groups*vec/32,
        'retire':ref_clock*phase['retire']['base_us']}
    if vf:
        stages['body']=ref_clock*phase['body']['base_us']+vf('post_mix',1,d,4)/c_per_us
    per_kib=READ_PER_KIB
    if chip.details.get('mte2_bandwidth_gbs'):
        per_kib=1024*chip.frequency_ghz/chip.details['mte2_bandwidth_gbs']
    lane=32*ceil(2*d/32)
    first,second=port['response_fixed_cycles']
    ready=max(first+lane*per_kib/1024,port['gap_cycles'][0]+second+4*lane*per_kib/1024)
    stages['read']=(ready+port['handoff_cycles'])/c_per_us
    stages['shared_link_floor']=max(0.,shared_floor-sum(stages.values()))
    return _scenario(stages,'channel_repeat_operand_response','v0.24 pre-CA channel-repeat schedule; distinct DT/PR shapes 8/7')


def _token_scenario(shape,chip,base,group,vf=None):
    previous=token_prices()
    state=group.split('_after_')[0]
    stage=dict(previous['by_state'][state])
    observed=_prices()['token_ordinary']
    control=observed['control'].get(group)
    if control and control['processes']>=2:
        stage['control_to_operand']=control['median_cycles']
    group_work=(previous['ready_to_write_per_group']*ceil(shape[2]/64)*(27/64)
                /(chip.vector_tops['32']/chip.vector_cores)*chip.frequency_ghz/1.65)
    for cohort in (group,'shared'):
        for k,line in observed['curves'].get(cohort,{}).items():
            if line:
                stage[k]=max(line['minimum'],line['fixed_compute_tail']+group_work)
    prices={**previous,'by_state':{state:stage}}
    pipe=token_pipeline(shape,chip,base['target_plan'],state,prices,base['shared_floor_us'],vf_price=vf)
    evidence=('v0.27 pre-CA ordinary token issue curves; small forward cohort'
              if control else 'shared steady-state curve and earlier state prices; previous entry lacks calibration')
    return _scenario(pipe['stages_us'],'token_operand_issue',evidence)


def estimate_post_history(shape,*,chip=REFERENCE_NAME,dtype=None) -> PostHistory:
    """Return conditional estimates for standalone BF16 [tokens,4,width] with hRes.

    The input is contiguous ND. Expected time requires a caller-specified prior.
    Adjacent identical means the last successfully dispatched kernel on the same
    rank and stream had the same shape, attributes, dtype and implementation.
    vLLM's separate HcPost implementation has a different contract.
    """
    target=load_chip(chip);_,chosen=_call_attrs('mhc_post',{},dtype)
    if chosen!='bf16':raise ValueError('Post history scenarios require BF16')
    shape=tuple(shape)
    _,_,active,base=resource_post(shape,target,{})
    estimated = _scenarios(shape, target, base)
    route = base['target_plan']['route']
    # Name alone cannot qualify edited clocks or resource counts as a measured 950 configuration.
    outside=target not in (load_chip(REFERENCE_NAME),load_chip('Ascend950PR_957b'))
    return PostHistory(shape,target.name,active,route,MappingProxyType(estimated),outside,target)


def _scenarios(shape, target, base,vf=None):
    """Conditional prices derived from shape, hardware and declared state contracts."""
    plan=base['target_plan'];route=plan['route']
    active=plan['active_aiv']
    phases=base['state_phases_us'];estimated={}
    v32=target.vector_tops['32']
    if route=='post_channel':
        for s in ('first','expand','reuse'):
            if s!='expand' or active>1:
                estimated[s]=_scenario(phases[s],'channel_resource','950 state-specific resource phases')
        if v32:
            estimated['repeat']=_channel_repeat(plan,target,base['shared_floor_us'],vf)
    else:
        groups=['first','reuse_after_channel','reuse_after_token']
        if active>1:groups.append('expand_after_channel')
        if active>ceil(target.vector_cores/2):groups.append('expand_after_token')
        tiles = channel_tiles(shape[2], target)
        for group in groups:
            state=group.split('_after_')[0]
            controls = _prices()['token_ordinary']['control'].get(group) if v32 else None
            if v32 and tiles and tiles['count'] == 2 and group == 'reuse_after_token':
                from operator_model.operators.mhc.fragment_pipeline import covered_fragment_schedule
                pipe = covered_fragment_schedule(target, base, tiles,vf_price=vf)
                estimated[group] = _scenario(pipe['stages_us'], pipe['model'],
                    'DT two-slot DMA schedule; limited paired geometry coverage')
            elif tiles and tiles['count'] == 1 and controls and controls['processes'] >= 2:
                estimated[group]=_token_scenario(shape,target,base,group,vf)
            else:
                estimated[group]=_scenario(phases[state],'occupancy_resource',
                    'No supported operand schedule for this predecessor, UB geometry or throughput')
        if v32 and tiles:
            pipe=token_pipeline(shape,target,plan,'repeat',token_prices(),base['shared_floor_us'],vf_price=vf)
            estimated['repeat']=_scenario(pipe['stages_us'],'token_immediate_repeat','frozen v0.25 token schedule, DT/PR independent repeat shapes 12/10')
    if 'repeat' not in estimated:
        estimated['repeat']=_scenario(phases['reuse'],'occupancy_resource','Target vector throughput or usable UB fragment size unavailable')
    return estimated


def price_post(shape, target, attrs):
    """Price a built-in Post call while retaining uncertainty about absent history.

    With no predecessor, a single point minimizes worst relative loss at the
    two modeled state extremes. A band of scenarios has no implied probability
    or guaranteed coverage of CA time.
    """
    result = resource_post(shape, target, attrs)
    if attrs.get('resident_aiv') is not None:
        return result

    _, aic, aiv, base = result
    scenarios = _scenarios(shape, target, base)
    state_times = {k:s.latency_us for k,s in scenarios.items()}
    low_state, high_state = min(state_times, key=state_times.get), max(state_times, key=state_times.get)
    low, high = state_times[low_state], state_times[high_state]
    high_weight = low/(low+high)
    point = 2*high*high_weight
    # Different mechanisms use different stage names. Giving them a common
    # fictitious timeline hides the uncertainty, so expose the two point contributions.
    stages = {'lower_scenario_component': (1-high_weight)*low,
              'upper_scenario_component': high_weight*high}
    return point, aic, aiv, {**base, 'model':'post_unknown_dispatch', 'stages_us':stages,
        'state_range_us':[low,high], 'state_scenarios_us':state_times,
        'endpoint_states':[low_state,high_state],
        'point_scope':'minimax relative error over modeled states; no execution-state probabilities',
        'range_scope':'conditional model envelope; individual CA windows may lie outside'}
