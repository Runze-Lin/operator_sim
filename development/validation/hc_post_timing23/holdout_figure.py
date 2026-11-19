"""Publish separate model/qualified-CA traces as their original independent receipts finish."""
import sys
import time

from extract import OUT, read
from figure import draw

CASES=(('holdout-Ascend950DT_9582-t4-d4080','dt-4080-independent'),
       ('holdout-Ascend950DT_9582-t33-d7168','dt-7168-independent'),
       ('supplement-Ascend950DT_9582-t9-d4160','dt-4160-supplement'))


if __name__=='__main__':
    for name,prefix in CASES:
        target=OUT/(name+'.json')
        while not target.exists():
            if '--watch' not in sys.argv:break
            time.sleep(9)
        if not target.exists() or not read(target)['qualified']:continue
        if not (OUT/(prefix+'.png')).exists():draw(target,prefix)
