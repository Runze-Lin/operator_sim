"""Launch a frozen Post batch. Reuse the audited v0.10 helper and CPU-reference gates."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
legacy = runpy.run_path(str(ROOT / 'scripts/wide-v10-collect.py'))
collect = legacy['main']
# The legacy collector hard-codes its scratch folder to runs/v10wide. Offset ids
# use a distinct range in the batch name; the immutable v0.10 batch is 0-11.
collect.__globals__['OUT'] = ROOT / 'results/v0.11/post'
if __name__ == '__main__':
    collect('mhc_post', 0, 10)
