"""Optional measured-port scenario for a single contiguous 950 FA reader."""
from dataclasses import dataclass
from functools import lru_cache

from operator_model.resources import load_parameters


@lru_cache(maxsize=1)
def _scenarios():
    return load_parameters('fa_port_scenarios')['skus']


@dataclass(frozen=True)
class FAPortScenario:
    evidence: dict

    @property
    def launch_cycles(self):
        return self.evidence['first_q_issue']

    @property
    def p_store_cycles(self):
        return self.evidence['first_p_store_span']

    @property
    def pv_issue_tail(self):
        return self.evidence['latest_p_retire_to_pv']

    def read(self, rows):
        """Return independent probe start interval and drained response, in cycles."""
        sample = self.evidence['nd2nz'].get(str(rows))
        return tuple(sample) if sample is not None else None

    def first_reads(self, q_rows, kv_rows, read_rate):
        """Completion relative to first Q issue. Port/bandwidth floors may overlap response."""
        q_probe = self.read(q_rows)
        page_probe = self.read(kv_rows)
        if q_probe is None or page_probe is None:
            raise ValueError('No matching independent ND2NZ probe for first Q or KV read')
        observed = self.evidence['first_response']
        # Queue conditions at entry differ from the independent drained probe.
        # Preserve this measured excess on the same template when Q rows change.
        q_response = observed[0] + q_probe[1] - self.read(9)[1]
        k_response = observed[1] + page_probe[1] - self.read(256)[1]
        v_response = observed[2] + page_probe[1] - self.read(256)[1]
        k_issue, v_issue = self.evidence['first_k_v_offsets']
        q_floor = 128 * q_rows / read_rate
        page_floor = 128 * kv_rows / read_rate
        return (max(q_floor, q_response), k_issue + max(page_floor, k_response),
                v_issue + max(page_floor, v_response))

    def page_interval(self, rows):
        """Two serial read issues share an engine; V→K control can elapse during V service."""
        sample = self.read(rows)
        if sample is None:
            return None
        issue, _ = sample
        return (2 * issue, issue + self.evidence['v_to_k_regular'])

    def first_softmax_ready(self, qk_ready, launch):
        """Relative to Q issue; observed first VF issue serves as a release lower bound."""
        return max(qk_ready, self.evidence['first_softmax_load_latest'] - launch)


def port_scenario(shape, chip, *, dtype='bf16'):
    """Get a matching explicitly requested profile; never implicitly reuse values on a new SKU."""
    if len(shape) == 3:
        shape = (*shape, 1, 1)
    if len(shape) != 5:
        raise ValueError('The FA probe scenario expects Q, KV, D, B, H')
    q, kv, d, b, h = shape
    evidence = _scenarios().get(chip.name)
    if evidence is None or chip.frequency_ghz != 1.65 or dtype != 'bf16':
        raise ValueError('FA probe requires 950DT_9582 or 950PR_957b at 1.65 GHz with native BF16')
    if (d, b, h) != (64, 1, 1) or not (2 <= q <= 64 and kv > 1024):
        raise ValueError('FA probe requires single-head contiguous Q64/KV256-template BF16, D64 and KV>1024')
    if str(q) not in evidence['nd2nz'] or chip.details.get('fa_aiv_per_aic', 2) != 2:
        raise ValueError('FA probe requires measured Q-row ND2NZ geometry and two AIVs per AIC')
    return FAPortScenario(evidence)
