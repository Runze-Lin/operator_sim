"""Scheduler-owned residency for one rank and one ordered stream of launches."""
from collections.abc import Mapping
from dataclasses import replace

from .backend import estimate
from .chips import REFERENCE_NAME, load_chip
from .predictor import ALIASES
from .rms_history import estimate_rms_history_band


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
    experimental_post_repeats enables the frozen per-row schedule for two
    adjacent, identical built-in BF16 Post launches in this stream.
    """

    def __init__(self, chip=REFERENCE_NAME, *, experimental_post_repeats=False):
        if not isinstance(experimental_post_repeats, bool):
            raise TypeError('experimental_post_repeats must be a boolean')
        self._post_repeats = experimental_post_repeats
        self.chip = load_chip(chip)
        self.reset()

    def reset(self):
        """Start an empty stream with a known cold first launch."""
        self._last_family = None
        self._resident = 0
        self._exact = True
        self._rms_prefix = None
        self._last_post = None

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

    def _prefix(self, family):
        if family == 'rmsnorm' and self._rms_prefix is not None:
            return self._rms_prefix
        if self._last_family in (None, family):
            return self._resident, self._exact
        return 0, False

    def _partial_rms(self, shape, options, dtype, forecast, resident):
        """Restrict a shape-only point to times allowed by the observed prefix."""
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

    def _with_history(self, family, op, shape, options, dtype):
        result = estimate(op, shape, options, chip=self.chip, dtype=dtype)
        if (self._last_post is not None and family == 'mhc_post' and
                result.latency_us is not None and self._last_post == (self.chip, tuple(shape))):
            from .post_repeat import price_repeat
            return price_repeat(result, shape, self.chip)
        return result

    def preview(self, op, shape, attrs=None, *, dtype=None):
        """Price a candidate without changing the next call."""
        family = _kernel_family(op, attrs)
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
        return self._with_history(family, op, shape, options, dtype)

    def advance(self, op, shape, attrs=None, *, dtype=None, aic_only=False):
        """Price a dispatch and record its AIV footprint on success.

        Set aic_only=True when the implementation's launch contract guarantees
        zero AIVs. The modeled footprint alone cannot establish this contract.
        """
        if not isinstance(aic_only, bool):
            raise TypeError('aic_only must be a boolean')
        result = self.preview(op, shape, attrs, dtype=dtype)
        if result.latency_us is None:
            return result
        family = _kernel_family(op, attrs)
        if aic_only and (not result.active_aic or result.active_aiv != 0):
            raise ValueError('aic_only conflicts with the modeled engine footprint')
        saved_rms = self._prefix('rmsnorm') if aic_only else None
        self._resident, self._exact = self._prefix(family)
        self._rms_prefix = saved_rms
        explicit = (attrs or {}).get('resident_aiv') if family in _STATEFUL else None
        if explicit is not None:
            self._resident = explicit
            self._exact = True
        self._resident = max(self._resident, result.active_aiv or 0)
        if self._resident == self.chip.vector_cores:
            self._exact = True
        self._last_family = family
        # Successful validation fixes BF16, contiguous ND, four streams and hRes for
        # the built-in kernel. Other kernels, including vLLM HcPost, clear this key.
        self._last_post = (self.chip, tuple(shape)) if self._post_repeats and family == 'mhc_post' else None
        return result
