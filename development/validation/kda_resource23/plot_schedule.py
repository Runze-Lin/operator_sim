"""Recover primitive budgets from the frozen single-head schedule for its comparison plot."""
import model

COLORS={ 'Shared phase':'#d3e0e7','MTE2':'#097f89','MTE1':'#5276ba','Cube':'#7850a7',
         'Vector VF':'#408b47','MTE3 / FIXP':'#bd582a','Front end':'#a9b7c1' }


class Recorder(model.Tile):
    """Instrument independent prices, preserving the original return value and ordering."""
    made=[]
    def __init__(self,*args):
        super().__init__(*args)
        self.calls=[]
        self.made.append(self)
    def dma(self,route,*sizes):
        t=super().dma(route,*sizes)
        engine='MTE2' if route.startswith('OUT') else 'MTE1' if route.startswith('L1') else 'MTE3 / FIXP'
        self.calls.append((route,engine,t))
        return t
    def calc(self,shapes):
        t=super().calc(shapes);self.calls.append(('MMAD','Cube',t));return t
    def vec(self,i,n):
        t=super().vec(i,n);self.calls.append(('VF','Vector VF',t));return t


def model_bars(index,pred,n=2,sku=model.DT):
    original=model.Tile
    model.Tile=Recorder
    try:
        chip=model.load_chip(sku);a=pred["active_aic"]
        tile=(lambda:model.prepare(n,chip,a),lambda:model.recurrence(n,chip,a,True,True),
              lambda:model.final(n,chip,a,False))[index]()
    finally:
        model.Tile=original
    calls=tile.calls; pos=0; rows={}; end=pred['first_payload_cycles']
    rows[0]=[(0,end/1650,'Front end')]
    def step(name,t,intervals):
        nonlocal pos
        actual,kind,duration=calls[pos]
        assert actual==name,(index,pos,actual,name)
        pos+=1
        intervals.append((t/1650,(t+duration)/1650,kind))
        return t+duration
    def walk(parts,t,intervals):
        for name in parts:
            if isinstance(name,tuple):
                t=max(step(p,t,intervals) for p in name)
            else:t=step(name,t,intervals)
        return t
    r='OUT->UB';l='OUT->L1';store='UB->OUT';u='L0C->UB';o='L0C->OUT'
    ab=('L1->L0A','L1->L0B')
    patterns=(
        [[r,'VF',store],['VF',store],[l,ab,'MMAD',u],['VF',store],None,[l,ab,'MMAD',o]],
        [[r,'VF',store],[l,ab,'MMAD',u],[r,'VF',store],None,['VF',store]],
        [[l],None,[o]]
    )
    for j,pattern in enumerate(patterns[index],start=1):
        events=[]; start=end
        if pattern:
            end=walk(pattern,start,events)
        elif index==0:
            end=max(walk([l],start,events),walk([r,'VF',store],start,events))
        elif index==1:
            end=max(walk([l,ab,'MMAD',u],start,events),walk([r],start,events))
        else:
            first=walk([ab],start,events)
            first_cube=walk(['MMAD'],first,events)
            second=walk([ab],first,events)
            end=walk(['MMAD'],max(first_cube,second),events)
        assert abs(end-start-tile.legs[j-1][1])<1e-5
        rows[j]=events
    assert pos==len(calls)
    assert abs(end/1650-pred['device_us'])<1e-8
    return rows
