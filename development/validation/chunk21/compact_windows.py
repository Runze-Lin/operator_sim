"""Count finalised component windows separately from the unfinished composite operator."""
import sqlite3
from probe import ROOT, RUN, ART, sha, save


def completed_db(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        info=dict(db.execute('SELECT KernelInfoAttr,KernelInfoVal FROM KernelInfo'))
        if 'exec_time_ns' not in info:
            return None
        count,lo,hi=db.execute('SELECT count(*),min(ExecInstrTickStart),max(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
    if info['instrs_exec_num']!=count or info['exec_time_ticks']!=hi-lo or info['frequency_ghz']!=1.65:
        raise ValueError('unverified compact helper window')
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),launch_id=info['launch_id'],
                helper_generic_name=info['name'],core_entries=info['ai_core_active'],
                count=count,elapsed_device_us=info['exec_time_ns']/1000,ticks=[lo,hi],frequency_ghz=info['frequency_ghz'])


if __name__=='__main__':
    entries=[]
    for tag in ('kimi-k3-bf16g-64x12-dt-compact','sub16'):
        dest=RUN/tag/'Ascend950DT_9582'
        windows=[row for p in sorted(dest.glob('profile/*/record/*.db')) if (row:=completed_db(p)) is not None]
        entries.append(dict(tag=tag,windows=windows,ca_total_us=None,qualification='individual closed helper windows; no verified full host completion'))
    save(ART/'partial-components.json',{'entries':entries})
