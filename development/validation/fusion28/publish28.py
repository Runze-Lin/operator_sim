"""Generate full-timing figures and Markdown only after each actual receipt is qualified."""
import os
import time
from landmarks import OUT,read
from figure28 import DT,PR,render
from forward28 import forward

def publish(watch=False):
    while True:
        for sku in (DT,PR):
            fn=OUT/'receipts'/('sequence-'+sku+'.json')
            name=OUT/f'independent-{sku[6:11].lower()}-t1153-i6.png'
            if fn.is_file() and not name.exists():
                row=next((r for r in read(fn)['rows'] if r['index']==2),None)
                if row and row['qualified']:render(sku,1153,True)
        forward()
        if not watch or read(OUT/'state.json')['state']=='finished':break
        time.sleep(30)
if __name__=='__main__':
    import sys
    publish('--watch' in sys.argv)
