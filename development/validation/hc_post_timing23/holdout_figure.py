"""Publish the two-page independent validation figure as soon as its full receipt is qualified."""
import sys
import time

from extract import OUT, read
from figure import draw


if __name__=='__main__':
    target=OUT/'holdout-Ascend950DT_9582-t7-d4112.json'
    while not target.exists():
        state=read(OUT/'holdout-state.json')
        if state['status'] in ('stopped','complete'):
            raise RuntimeError('Registered figure case has no qualified receipt')
        if '--watch' not in sys.argv:raise FileNotFoundError(target)
        time.sleep(10)
    if not read(target)['qualified']:raise ValueError('Failed golden or incomplete timing')
    draw(target,'dt-4112-independent')
