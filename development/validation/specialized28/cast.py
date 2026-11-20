"""Register and audit the frozen streaming Cast alternatives without tuning to a SKU."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.28/cast'
PAIR = ROOT / 'results/v0.27/memory/cast-pair'
PR, DT = 'Ascend950PR_957b', 'Ascend950DT_9582'


def read(path):
    return json.loads(path.read_bytes())


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def register():
    future = read(PAIR/'freeze.json')
    terminal = list(PAIR.glob('*-result.json'))
    if terminal:
        raise ValueError('New selection rule must precede either paired timing receipt')
    sources = [ROOT/'development/validation/cache25/shadow.py',ROOT/'results/v0.25/cache/replay.json',
               ROOT/'results/v0.26/memory/pro-256ki-prefix-forecast.json',PAIR/'freeze.json']
    data = dict(utc=datetime.now(timezone.utc).isoformat(),hypothesis='Cast uses 2 B read and 4 B dirty store per conversion. A slow writeback relative to aggregate active-AIV issue can leave proportionally more dirty output resident as clean input is consumed. This is an occupancy hypothesis, not a measured hit rate.',
                formula='R=min(L,X+Y); q=Y/(X+Y); rho=min(1,B_ext/I_AIV); D=min(Y,R-R*(1-q)*rho); external=X+Y-D. B and I in GB/s; D,R,X,Y,L in bytes. Previously published uniform policy sets rho=1.',
                data_roles='Legacy replay, 156 MiB complete DT Cast, complete 256Ki Pro and companion Sigmoid are known before this selection. 165 MiB pair pricing uses the preceding independent freeze; no outcome/dump from the in-progress pair used for selector design.',
                promotion=['Entire preregistered new pair must pass original ELF, submitted geometry, all output values and all-core endpoint checks.',
                           'Pressure must beat old absolute whole-window error on the PR new shape; both new endpoints must have <=25% APE.',
                           'Historical DT and PR each keep WAPE<=25% and max<=50% in every published geometric subgroup after applying the single formula.',
                           'Both 256Ki endpoints stay within 25% absolute error. Report any regression, including speed ratio and in-tile timing. If evidence is insufficient, keep default and expose mechanism as sensitivity.'],
                prices=future['jobs'][0]['predictions'], sha256={str(p.relative_to(ROOT)):checksum(p) for p in sources})
    with (OUT/'register.json').open('x') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def score():
    reg=read(OUT/'register.json')
    for name,expected in reg['sha256'].items():
        if checksum(ROOT/name)!=expected:
            raise ValueError('changed frozen source: '+name)
    old=read(ROOT/'results/v0.25/cache/replay.json')
    historical=[r for r in old['groups'] if r['op']=='cast_bf16_fp32']
    completed={}
    for chip,tag in ((PR,'pr'),(DT,'dt')):
        path=PAIR/f'cast_bf16_fp32-{tag}-result.json'
        if path.exists():
            record=read(path)
            if record['qualified'] and record['forecast']!=reg['prices'][chip]:
                raise ValueError('changed candidate')
            completed[chip]={'qualified':record['qualified'],'ca_us':record.get('ca_us'),
                             'error_pct':record.get('error_pct'),'wall_s':record['cost']['seconds'],
                             'reason':record.get('reason'),'receipt':str(path.relative_to(ROOT))}
    large={c:read(ROOT/f'results/v0.26/memory/pro-b8-256ki-{t}-score.json')['metrics'] for c,t in ((DT,'dt'),(PR,'pr'))}
    measured=len(completed)==2 and all(r['qualified'] for r in completed.values())
    passed=False
    ratios=None
    if measured:
        e=completed[PR]['error_pct']
        passed=(abs(e['pressure'])<abs(e['uniform']) and
                all(abs(r['error_pct']['pressure'])<=25 for r in completed.values()) and
                all(r['pressure']['wape']<=25 and r['pressure']['max_absolute_pct']<=50 for r in historical) and
                all(abs(r['pressure']['full_pct'])<=25 for r in large.values()))
        ratios={k:reg['prices'][DT][k]['us']/reg['prices'][PR][k]['us'] for k in ('uniform','pressure','clean_first')}
        ratios['ca']=completed[DT]['ca_us']/completed[PR]['ca_us']
    out=dict(freeze_sha256=checksum(OUT/'register.json'),historical=historical,
             heldout=completed,large=large,paired_dt_over_pr=ratios,fully_qualified=measured,promotion_passed=passed)
    (OUT/'score.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    args=ArgumentParser(description=__doc__)
    args.add_argument('action',choices=['register','score'])
    {'score':score,'register':register}[args.parse_args().action]()
