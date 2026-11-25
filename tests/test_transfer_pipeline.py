import unittest
from dataclasses import asdict, replace
from itertools import product

from operator_model import estimate, load_chip
from operator_model.scheduling.transfer_pipeline import (
    TransferProfile,
    replay_program,
    replay_tiles,
    stream_bound,
)


def profile(direction, ii, response):
    return TransferProfile(
        "copy", direction, "ND contiguous 32 BF16", "bf16", 64, ii, response, "synthetic test"
    )


class TransferStreams(unittest.TestCase):
    def test_issue_drained_and_slot_constraints(self):
        for n, slots, interval, compute in product((1, 2, 5, 9), (1, 2, 4), (2, 8), (0, 7)):
            read, write = profile("GM→UB", interval, 17), profile("UB→GM", 4, 10)
            chip = replace(load_chip(), transfer_profiles={"r": asdict(read), "w": write})
            args = dict(tiles=n, slots=slots, compute_cycles=compute, read_profile="r", write_profile="w")
            last = list(replay_tiles(read, write, compute, n, slots))[-1]
            self.assertEqual(stream_bound(chip, args)["cycles"], last["write_retire"])

    def test_tail_does_not_use_full_tile_response(self):
        full, short = profile("GM→UB", 4, 20), profile("GM→UB", 2, 4)
        write = profile("UB→GM", 3, 6)
        events = list(replay_program([(full, write, 5), (full, write, 5), (short, write, 1)], 2))
        self.assertEqual(events[-1]["read_ready"] - events[-1]["read_issue"], 4)
        self.assertGreaterEqual(events[-1]["read_issue"], events[0]["write_retire"])

    def test_api_applies_one_floor_without_double_counting(self):
        chip = replace(
            load_chip(), transfer_profiles={"r": profile("GM→UB", 5, 8000), "w": profile("UB→GM", 3, 5500)}
        )
        for op in ("cast_bf16_fp32", "rmsnorm"):
            a = dict(tiles=5, slots=2, compute_cycles=16, read_profile="r", write_profile="w")
            base = estimate(op, [64, 2048], chip=chip)
            result = estimate(op, [64, 2048], {"transfer_pipeline": a}, chip=chip)
            self.assertAlmostEqual(result.latency_us, max(base.latency_us, stream_bound(chip, a)["bound_us"]))
            self.assertIsNone(estimate(op, [64, 2048], {"transfer_pipeline": a}).latency_us)

    def test_stream_applies_bound_after_history(self):
        from operator_model import OperatorStream

        chip = replace(
            load_chip(), transfer_profiles={"r": profile("GM→UB", 8, 50000), "w": profile("UB→GM", 5, 25000)}
        )
        attrs = dict(tiles=2, slots=1, compute_cycles=10, read_profile="r", write_profile="w")
        stream = OperatorStream(chip)
        for _ in range(2):
            result = stream.advance("mhc_post", [1, 4, 4096], {"transfer_pipeline": attrs})
            self.assertAlmostEqual(result.latency_us, stream_bound(chip, attrs)["bound_us"])
