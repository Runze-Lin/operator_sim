"""Refresh only the two-slot scorer while the already registered Cast jobs run."""
import time
from prospective import OUT, read, score

if __name__=='__main__':
    while True:
        score()
        if not read(OUT/'two-slot-scores.json')['pending']:
            break
        time.sleep(45)
