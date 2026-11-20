"""Score incoming registered cache receipts and plot the PR windows once they qualify."""
import fcntl
import json
from pathlib import Path
import time

from block_figure import draw as draw_cast, load_blocks
from fa_tail import SUP, collect, draw as draw_fa
from forward import CACHE, OUT, REG, ROOT, read, score


def run():
    tmp = ROOT/'runs/memory27-followup'
    tmp.mkdir(exist_ok=True)
    with (tmp/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous = None
        while True:
            ca = ROOT/'results/v0.23/pro-prefill/pro-b8-256ki-pr.json'
            stop = ROOT/'results/v0.26/memory/pro-b8-256ki-pr-score.json'
            fa = SUP/'physical/fa-supp-Ascend950PR_957b-9-32769-64-1-1.json'
            failures = tmp/'errors.json'
            errors = read(failures) if failures.exists() else {}
            dest = OUT.parent/'fa-32769-pr-tail.json'
            if fa.exists() and not dest.exists() and 'fa' not in errors:
                if any(r['id']==fa.stem for r in read(SUP/'score.json')['rows']):
                    try:
                        d = collect('Ascend950PR_957b')
                        draw_fa(d, dest.with_suffix('.png'))
                        dest.write_text(json.dumps(d, indent=2)+'\n')
                    except (ValueError, AssertionError, IndexError, KeyError, OSError) as exc:
                        errors['fa']=repr(exc)
            cast_plot = OUT/'pro-256ki-pr-tail.png'
            if stop.exists() and not cast_plot.exists() and 'cast' not in errors and read(stop)['qualified']:
                try:
                    d=load_blocks('pr')
                    draw_cast(d,len(d['ca_tiles'])-3,4,cast_plot)
                except (ValueError, AssertionError, IndexError, KeyError, OSError) as exc:
                    errors['cast']=repr(exc)
            if errors:
                failures.write_text(json.dumps(errors,indent=2)+'\n')
            ids=[c['id'] for c in read(REG)['cases']]
            returned=tuple(i for i in ids if (CACHE/(i+'-result.json')).is_file())
            if returned != previous:
                score()
                previous=returned
            cache_done=len(returned)==len(ids) or any(not read(CACHE/(i+'-result.json'))['qualified'] for i in returned)
            fa_status=read(SUP/'state.json')['state']
            fa_done=dest.exists() or 'fa' in errors or fa_status.startswith(('finished','complete','failed','stopped'))
            cast_done=cast_plot.exists() or 'cast' in errors or (ca.exists() and not read(ca)['qualified'])
            if cache_done and fa_done and cast_done:
                break
            time.sleep(45)


if __name__=='__main__':
    run()
