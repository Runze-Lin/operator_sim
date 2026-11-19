"""Scheduler-owned residency for one rank and one ordered stream of launches."""
from collections.abc import Mapping

from .backend import estimate
from .chips import REFERENCE_NAME, load_chip
from .predictor import ALIASES


_STATEFUL = {'rmsnorm', 'mhc_post'}


class OperatorStream:
    """Track AIV residency when the order of successfully dispatched calls is known.

    A switch of kernel family discards the exact high-water mark. Successive
    launches in the new family can still establish a lower bound on warm AIVs.
    """

    def __init__(self, chip=REFERENCE_NAME):
        self.chip = load_chip(chip)
        self.reset()

    def reset(self):
        """Start an empty stream with a known cold first launch."""
        self._last_family = None
        self._resident = 0
        self._exact = True

    def preview(self, op, shape, attrs=None, *, dtype=None):
        """Price a candidate without changing the next call."""
        family = ALIASES.get(op, op) if isinstance(op, str) else None
        if family not in _STATEFUL or (attrs is not None and not isinstance(attrs, Mapping)):
            return estimate(op, shape, attrs, chip=self.chip, dtype=dtype)
        options = dict(attrs or {})
        continuing = self._last_family in (None, family)
        if 'resident_aiv' in options or not continuing:
            return estimate(op, shape, options, chip=self.chip, dtype=dtype)
        if not self._exact:
            forecast = estimate(op, shape, options, chip=self.chip, dtype=dtype)
            if forecast.active_aiv is None or forecast.active_aiv > self._resident:
                return forecast
        options['resident_aiv'] = self._resident
        return estimate(op, shape, options, chip=self.chip, dtype=dtype)

    def advance(self, op, shape, attrs=None, *, dtype=None):
        """Price a dispatch and record its AIV footprint on success."""
        result = self.preview(op, shape, attrs, dtype=dtype)
        if result.latency_us is None:
            return result
        family = ALIASES.get(op, op)
        explicit = (attrs or {}).get('resident_aiv') if family in _STATEFUL else None
        if explicit is not None:
            self._resident = explicit
            self._exact = True
        elif self._last_family not in (None, family):
            self._resident = 0
            self._exact = False
        self._resident = max(self._resident, result.active_aiv or 0)
        if self._resident == self.chip.vector_cores:
            self._exact = True
        self._last_family = family
        return result
