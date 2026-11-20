"""Recheck the same pre-registered queries with a functional simulator domain socket."""
import json
import host_audit as audit

if __name__ == '__main__':
    old = [json.loads((audit.OUT/(soc+'-workspace.json')).read_bytes()) for soc in audit.SOCS]
    assert all(not x['qualified_for_host_choice'] and x['observed_soc'] != x['soc'] for x in old)
    audit.RAW = audit.RAW/'host-socket'
    audit.OUT = audit.OUT/'host-socket'
    audit.RAW.mkdir(exist_ok=True); audit.OUT.mkdir(exist_ok=True)
    for soc in audit.SOCS:
        if not (audit.OUT/(soc+'-workspace.json')).exists(): audit.collect(soc)
