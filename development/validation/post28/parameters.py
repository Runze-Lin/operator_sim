"""Package previously frozen stage prices without fitting any v0.28 output."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28/post/parameters.json'
CHANNEL='results/v0.24/post/freeze.json'
ISSUE='results/v0.27/post/parameters.json'
TOKEN='results/v0.25/post/prices.json'


def bundle():
    source=lambda f:json.loads((ROOT/f).read_text())
    return dict(description='Original pre-CA operand and issue prices. Distinct channel-repeat and token routes; no v0.28 fit.',
        sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in (CHANNEL,ISSUE,TOKEN,'development/validation/post24/model.py','development/validation/post27/issue.py','operator_model/post_token_pipeline.py')},
        channel_repeat=source(CHANNEL)['parameters']['post_channel']['repeat'],token_ordinary=source(ISSUE))


if __name__=='__main__':
    p=bundle()
    if OUT.exists() and json.loads(OUT.read_text())!=p:
        raise RuntimeError('frozen parameters changed')
    if not OUT.exists():
        OUT.write_text(json.dumps(p,indent=2)+'\n')
