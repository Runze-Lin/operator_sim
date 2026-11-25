"""Scheduler-owned residency for one rank and one ordered stream of launches."""
from collections.abc import Mapping
from dataclasses import replace

from operator_model.backend import _memory_window, estimate
from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.scheduling.reference import ALIASES
from operator_model.operators.rmsnorm.history import estimate_rms_history_band


_STATEFUL = {'rmsnorm', 'mhc_post'}


def _kernel_family(op, attrs):
    family = ALIASES.get(op, op) if isinstance(op, str) else None
    if family == 'mhc_post' and isinstance(attrs, Mapping) and attrs.get('implementation') == 'vllm_hc_post':
        return 'vllm_hc_post'
    return family


class OperatorStream:
    """Track AIV residency when the order of successfully dispatched calls is known.

    Successive launches establish a lower bound on warm AIVs. A verified
    AIC-only dispatch can carry the RMSNorm prefix across a family switch.
    Adjacent identical built-in BF16 Post launches use the calibrated token
    schedule. Set experimental_post_repeats=False to replay the earlier
    occupancy-only estimate. The flag name is retained for existing callers.
    Set track_sinkhorn_history=True to reconstruct the standalone FP32 Sinkhorn
    high-water mark from eligible launches on this stream. After a family switch
    or unknown prefix, only sufficient same-family coverage selects a warm call.
    By default, an observed first launch or shape change can
    select the frozen operand-issue schedule for a single-fragment token row.
    A covered token-to-token change with two UB fragments uses the two-slot
    schedule. Disable experimental_post_fragments to reproduce whole-row pricing.
    """

    def __init__(self, chip=REFERENCE_NAME, *, experimental_post_repeats=True,
                 track_sinkhorn_history=False, track_post_history=True, experimental_post_fragments=None,
                 memory_policy="ca_window"):
        if not isinstance(experimental_post_repeats, bool):
            raise TypeError('experimental_post_repeats must be a boolean')
        if not isinstance(track_sinkhorn_history, bool):
            raise TypeError('track_sinkhorn_history must be a boolean')
        if not isinstance(track_post_history, bool):
            raise TypeError('track_post_history must be a boolean')
        if experimental_post_fragments is None:
            experimental_post_fragments = track_post_history
        if not isinstance(experimental_post_fragments, bool):
            raise TypeError('experimental_post_fragments must be a boolean')
        if memory_policy not in ("ca_window", "cold_io"):
            raise ValueError("memory_policy must be ca_window or cold_io")
        self.memory_policy = memory_policy
        self._post_fragments = experimental_post_fragments
        self._post_repeats = experimental_post_repeats
        self._track_sinkhorn = track_sinkhorn_history
        self._track_post = track_post_history
        self.chip = load_chip(chip)
        self.reset()

    def reset(self):
        """Start an empty stream with a known cold first launch."""
        self._last_family = None
        self._resident = 0
        self._exact = True
        self._rms_prefix = None
        self._last_post = None
        self._post_route = None

    def forget_history(self):
        """Attach to a running stream, or record an unmodeled completed dispatch.

        No predecessor coverage is known at this point. Further launches of
        the same family can establish a new lower bound without profiler input.
        """
        self._last_family = None
        self._resident = 0
        self._exact = False
        self._rms_prefix = None
        self._last_post = None
        self._post_route = None

    def _prefix(self, family):
        if family == 'rmsnorm' and self._rms_prefix is not None:
            return self._rms_prefix
        if self._last_family in (None, family):
            return self._resident, self._exact
        return 0, False

    def _partial_rms(self, shape, options, dtype, forecast, resident):
        """Restrict a shape-only point to times allowed by the observed prefix."""
        if (forecast.projection or {}).get('vector_model') == 'instruction_classes':
            # Only a lower bound on predecessor residency is known. Evaluate two
            # physical endpoints with the requested graph; no legacy CA band is
            # a bound on this alternative. Intermediate prefixes may lie outside.
            endpoints = [estimate('rmsnorm', shape, {**options, 'resident_aiv': h},
                                  chip=self.chip, dtype=dtype) for h in (resident,forecast.active_aiv)]
            if any(e.latency_us is None for e in endpoints):
                return forecast
            lo,hi=sorted(e.latency_us for e in endpoints)
            point=2*lo*hi/(lo+hi)
            from operator_model.backend import _rates
            context=dict(minimum_resident_aiv=resident, sampled_prefixes=[resident,forecast.active_aiv],
                         sampled_endpoint_range_us=[lo,hi], point_rule='relative minimax of two modeled endpoints')
            projections=dict(forecast.projection,latency_us=point,partial_history=context,
                             stages_us={'partial_history_center':point},timing_diagnostics={'partial_history':context})
            return replace(forecast,latency_us=point,projection=projections,rates=_rates(forecast.work,point,self.chip))
        band = estimate_rms_history_band(shape, chip=self.chip, dtype=dtype)
        times = band.by_resident_aiv_us[resident:]
        lower, upper = min(times), max(times)
        point = forecast.latency_us
        selected = None
        if point < lower or point > upper:
            endpoint = lower if point < lower else upper
            selected = resident + times.index(endpoint)
            forecast = estimate('rmsnorm', shape, {**options, 'resident_aiv': selected},
                                chip=self.chip, dtype=dtype)
        context = dict(minimum_resident_aiv=resident, possible_us=[lower, upper],
                       shape_only_us=point, endpoint_scenario_aiv=selected)
        return replace(forecast, projection={**forecast.projection, 'partial_history': context})

    def _with_history(self, family, op, shape, options, dtype, *, infer_post=False):
        result = estimate(op, shape, options, chip=self.chip, dtype=dtype)
        if family == 'mhc_post' and result.latency_us is not None:
            from operator_model.operators.vector.family_programs import Pricer
            vf=Pricer(self.chip) if (result.projection or {}).get('vector_model')=='instruction_classes' else None
            def finalize(priced):
                if vf is None or not vf.samples:
                    return priced
                d=dict(priced.projection.get('timing_diagnostics',{}),vector_model='instruction_classes',instruction_samples=vf.samples)
                return replace(priced,projection=dict(priced.projection,timing_diagnostics=d,instruction_samples=vf.samples),
                               reason='Observed Post history, with instruction-class SIMD service on the ordered schedule')
            if self._last_post == (self.chip, tuple(shape)):
                if self._post_repeats:
                    from operator_model.operators.mhc.repeat import price_repeat
                    return finalize(price_repeat(result, shape, self.chip,vf))
            elif infer_post:
                if self._post_fragments and self._post_route == 'post_token':
                    from operator_model.operators.mhc.ordered import price_ordered_fragments
                    priced = price_ordered_fragments(result, shape, self.chip, options['resident_aiv'], self._last_post[1],vf)
                    if priced is not None:
                        return finalize(priced)
                if self._track_post:
                    from operator_model.operators.mhc.ordered import price_ordered_token
                    return finalize(price_ordered_token(result, shape, self.chip, self._post_route,vf))
        return result

    def _sink_dispatch(self, op, shape, attrs, dtype):
        """Return a price and the family whose coverage this dispatch can prove."""
        options = dict(attrs or {})
        resident, exact = self._prefix('mhc_sinkhorn')
        explicit = 'resident_aiv' in options
        h = options.get('resident_aiv', resident)
        # The backend alone checks the layout, UB, dtype, iterations and target
        # resources. Its active AIV count also covers valid one-block large T
        # outside the shape-only model's domain.
        conditional = estimate(op, shape, {**options, 'resident_aiv': 0 if h is None else h},
                               chip=self.chip, dtype=dtype)
        if explicit and h is not None:
            return conditional, 'mhc_sinkhorn'
        compatible = conditional.latency_us is not None
        if compatible and not explicit and (exact or resident >= conditional.active_aiv):
            if not exact:
                # Only a lower bound is known, but every feasible history
                # selects the same resident schedule for this activity level.
                context = {'minimum_resident_aiv': resident, 'maximum_resident_aiv': self.chip.vector_cores}
                conditional = replace(conditional, projection={**conditional.projection, 'partial_history': context})
            return conditional, 'mhc_sinkhorn'
        fallback = estimate(op, shape, options, chip=self.chip, dtype=dtype)
        # An older fallback may still return a time on hardware with no
        # calibrated local price. It cannot establish same-kernel residency.
        return fallback, 'mhc_sinkhorn' if compatible else 'untracked_sinkhorn'

    def _dispatch(self, op, shape, attrs, dtype):
        if isinstance(attrs, Mapping) and 'transfer_pipeline' in attrs:
            from operator_model.backend import _transfer_window
            options = dict(attrs)
            program = options.pop('transfer_pipeline')
            result, family = self._dispatch(op, shape, options, dtype)
            return (_transfer_window(result, self.chip, program) if program is not None else result), family
        family = _kernel_family(op, attrs)
        if self._track_sinkhorn and family == 'mhc_sinkhorn' and (attrs is None or isinstance(attrs, Mapping)):
            return self._sink_dispatch(op, shape, attrs, dtype)
        return self._preview_existing(family, op, shape, attrs, dtype), family

    def preview(self, op, shape, attrs=None, *, dtype=None):
        """Price a candidate without changing the next call."""
        return _memory_window(self._dispatch(op, shape, attrs, dtype)[0], self.chip, self.memory_policy)

    def _preview_existing(self, family, op, shape, attrs, dtype):
        if family not in _STATEFUL or (attrs is not None and not isinstance(attrs, Mapping)):
            return estimate(op, shape, attrs, chip=self.chip, dtype=dtype)
        options = dict(attrs or {})
        if 'resident_aiv' in options:
            return self._with_history(family, op, shape, options, dtype)
        resident, exact = self._prefix(family)
        if not exact:
            forecast = estimate(op, shape, options, chip=self.chip, dtype=dtype)
            if (family == 'rmsnorm' and forecast.latency_us is not None and
                    forecast.active_aiv is not None and 0 < resident < forecast.active_aiv):
                return self._partial_rms(shape, options, dtype, forecast, resident)
            if forecast.active_aiv is None or forecast.active_aiv > resident:
                return forecast
        options['resident_aiv'] = resident
        return self._with_history(family, op, shape, options, dtype, infer_post=True)

    def advance(self, op, shape, attrs=None, *, dtype=None, aic_only=False):
        """Price a dispatch and record its AIV footprint on success.

        Set aic_only=True when the implementation's launch contract guarantees
        zero AIVs. The modeled footprint alone cannot establish this contract.
        """
        if not isinstance(aic_only, bool):
            raise TypeError('aic_only must be a boolean')
        result, family = self._dispatch(op, shape, attrs, dtype)
        if result.latency_us is None:
            return result
        if aic_only and (not result.active_aic or result.active_aiv != 0):
            raise ValueError('aic_only conflicts with the modeled engine footprint')
        saved_rms = self._prefix('rmsnorm') if aic_only else None
        self._resident, self._exact = self._prefix(family)
        self._rms_prefix = saved_rms
        tracked_sink = self._track_sinkhorn and family == 'mhc_sinkhorn'
        options = attrs or {}
        explicit = options.get('resident_aiv') if family in _STATEFUL or tracked_sink else None
        if ((tracked_sink or ((self._track_post or self._post_fragments) and family == 'mhc_post'))
                and 'resident_aiv' in options and explicit is None):
            # The caller declared an unknown predecessor. The completed launch
            # still establishes a lower bound for the following same-family one.
            self._resident, self._exact = 0, False
        if explicit is not None:
            self._resident = explicit
            self._exact = True
        self._resident = max(self._resident, result.active_aiv or 0)
        if self._resident == self.chip.vector_cores:
            self._exact = True
        self._last_family = family
        # Successful validation fixes BF16, contiguous ND, four streams and hRes for
        # the built-in kernel. Other kernels, including vLLM HcPost, clear this key.
        track_route = self._track_post or self._post_fragments
        track_post = family == 'mhc_post' and (self._post_repeats or track_route)
        self._last_post = (self.chip, tuple(shape)) if track_post else None
        self._post_route = result.projection['target_plan']['route'] if track_route and track_post else None
        return _memory_window(result,self.chip,self.memory_policy)
