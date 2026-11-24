"""Score the current packed consumer on qualified native static-metadata windows."""
import json
from pathlib import Path
from operator_model import estimate

HERE = Path(__file__).resolve().parent


def score(label, shape, cap, columns, sku, ca, frozen):
    attrs = dict(implementation="a5_qsli_projection",candidate_limit=cap,table_pages=columns)
    def quote(chip, policy="ca_window"):
        return estimate("candidate_filtered_indexer",shape,attrs,dtype="mxfp4",chip=chip,memory_policy=policy)
    e, cold, future = quote(sku),quote(sku,"cold_io"),quote("Ascend970DT")
    return dict(case=label,shape=shape,attrs=attrs,sku=sku,ca_us=ca,frozen_us=frozen,
                model_us=e.latency_us,signed_error_pct=100*(e.latency_us/ca-1),
                logical_roofline_factor=e.rates["logical_roofline"]["time_factor"],
                cold_U_pct=100*cold.rates["roofline"]["utilization"],
                future_970DT_us=future.latency_us)


def score_windows():
    def receipt(file):
        return json.loads((HERE/file).read_text())
    cases = [score("compact or wide PA", (1,1,32,128,1024,2048,512),81,r["table_columns"],r["sku"],
                   r["exact_native_kernel_window_us"],r["precollection_prediction_us"])
             for r in receipt("fused-qsli.json")]
    for r in receipt("long-row-results.json")["records"]:
        cases.append(score("1Mi full candidate row",(1,1,32,128,1048576,2048,512),2048,8201,r["chip"],
                           r["ca_us"],r["model_us"]))
    wave = receipt("prefill-wave.json")
    cases.append(score("frozen independent 32-query wave", tuple(wave["shape"]), 2048, 8201,
                       wave["sku"], wave["device_us"], wave["interval_prediction_us"]))
    return cases


if __name__=="__main__":
    (HERE/"qsli-evaluation.json").write_text(json.dumps(score_windows(),indent=2)+"\n")
