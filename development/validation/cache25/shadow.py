"""Offline scenario for the external memory of a cold, contiguous, single-pass stream."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.cast_resource import resource_cast
from operator_model.sigmoid_resource import resource_sigmoid
from operator_model.vector_memory import stream_memory


def price(chip,active,read_bytes,write_bytes,policy):
    m=stream_memory(chip,active,read_bytes,write_bytes)
    total=read_bytes+write_bytes
    l2=chip.details.get('l2_bytes') or 0
    # Only space occupied by this working set matters.
    occupied=min(l2,total)
    q=write_bytes/total
    pressure=min(1,chip.hbm_bandwidth_gbs/m['external_issue_gbs'])
    if policy=='clean_first':  # an admissible boundary; reserve no finished input
        dirty=min(occupied,write_bytes)
    elif policy=='pressure':  # old sensitivity, now tested on independent cross-operator shapes
        clean_reserve=occupied*(1-q)*pressure
        dirty=min(write_bytes,occupied-clean_reserve)
    elif policy=='uniform':
        dirty=m['assumed_retained_write_bytes']
    else:
        raise ValueError(policy)
    external=total-dirty
    return {**m,'l2_input_reserve_bytes':occupied-dirty,'assumed_retained_write_bytes':dirty,
            'modeled_external_write_bytes':write_bytes-dirty,'modeled_external_bytes':external,
            'external_floor_cycles':external*chip.frequency_ghz/m['effective_external_gbs'],
            'bandwidth_to_mixed_issue_ratio':pressure}


def predict(op,shape,sku):
    prop=load_chip(sku)
    route={'cast_bf16_fp32':resource_cast,'sigmoid_fp32':resource_sigmoid}[op]
    scenarios={}
    for policy in ('uniform','pressure','clean_first'):
        # Local single-process offline caller; the installable model and simulator are unmodified.
        with patch('operator_model.vector_memory.stream_memory',lambda *a: price(*a,policy)):
            t,_,active,m=route(shape,prop)
        scenarios[policy]=dict(us=t,active_aiv=active,external_GB_s=m['effective_external_gbs'],
            mixed_issue_GB_s=m['external_issue_gbs'],retained_output_MiB=m['assumed_retained_write_bytes']/2**20,
            external_read_plus_write_MiB=m['modeled_external_bytes']/2**20,external_write_MiB=m['modeled_external_write_bytes']/2**20,
            device_L2_MiB=(prop.details.get('l2_bytes') or 0)/2**20)
    return scenarios
