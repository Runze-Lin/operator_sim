"""Adapt one rank's ordered operator launches to a simulator clock in seconds."""
from operator_model import Estimate, OperatorStream


def duration_seconds(result: Estimate, *, allow_extrapolation: bool) -> float:
    if result.scheduled_us is None:
        raise ValueError(f"Cannot schedule this operator: {result.status} ({result.reason})")
    if result.status == "extrapolated" and not allow_extrapolation:
        raise ValueError("This simulation requires a calibrated target")
    return result.scheduled_us * 1e-6


def main():
    stream = OperatorStream("Ascend950DT_9582")
    elapsed_s = 0.0
    # These launches have an explicit dependency on the previous launch.
    for op, shape in [("matmul", (1280, 2048, 512)), ("rmsnorm", (1280, 2048))]:
        preview = stream.preview(op, shape, dtype="bf16")
        duration_seconds(preview, allow_extrapolation=True)
        issued = stream.advance(op, shape, dtype="bf16")
        elapsed_s += duration_seconds(issued, allow_extrapolation=True)
        print(f"{op:8s} {issued.latency_us:8.3f} µs  {issued.status}  "
              f"AIC={issued.active_aic} AIV={issued.active_aiv}")
    print(f"Dependency chain finishes at {elapsed_s:.9f} s")


if __name__ == "__main__":
    main()
