import numpy as np, pandas as pd, itertools, json, sys
from sysA import *
def sig2(F, D, s):
    # yön ayrık eşikli 1D aşırı uzama dönüşü (+ opsiyonel OI/hacim/BTC ön koşulu)
    H = s['H']; c = D['c']; dP = (c / np.r_[np.full(H, np.nan), c[:-H]] - 1) * 100
    L = dP <= -s['aL']; S_ = dP >= s['aS']
    return np.nan_to_num(np.where(L, 1, np.where(S_, -1, 0))).astype(np.int64)
def mot2(D, F, s):
    if s['fam'] == 'PXA': return g_ent(D['c'], sig2(F, D, s), int(s['hold']), s['sl'])
    return mot(D, F, s)
def sysR2(D, F, order, S): return simR(D, chain([mot2(D, F, S[m]) for m in order]))
GR = dict(PXA=dict(H=[192, 240, 288, 336, 432], aL=[3.5, 4, 4.5, 5, 5.5, 6, 7], aS=[3.5, 4, 4.5, 5, 5.5, 6, 7], hold=[144, 192, 240, 288, 336, 432, 576], sl=[5, 8, 10, 15, 20]),
          OIK=dict(a=[3, 3.5, 4, 4.5, 5], b=[3, 4, 5, 6, 8], hold=[144, 288, 432]),
          HVX=dict(a=[3, 3.5, 4, 4.5, 5], b=[1.2, 1.3, 1.45, 1.6, 2], hold=[144, 288, 432]),
          REL=dict(a=[4, 5, 6, 7, 8], hold=[144, 288, 432]),
          Z96=dict(a=[3, 3.5, 4], b=[0, 0.5, 1], sl=[5, 8], hold=[144, 288]))
def cd(D, F, order, S, rounds=2, log=True):
    best = metrics(D, sysR2(D, F, order, S)); bs = sc(best)
    for r in range(rounds):
        for m in order:
            for k in GR[m]:
                for v in GR[m][k]:
                    T = json.loads(json.dumps(S)); T[m][k] = v; x = metrics(D, sysR2(D, F, order, T))
                    if sc(x) > bs: bs = sc(x); best = x; S = T
            if log: print(r, m, S[m], {k: best[k] for k in KS}, flush=True)
    return S, best
if __name__ == '__main__':
    D = cut(load(5), SPLIT); F = feats(D); S0 = json.loads(json.dumps(CAND)); S0['PXA'] = dict(fam='PXA', H=288, aL=5.0, aS=5.0, hold=288, sl=10)
    S, b = cd(D, F, ['PXA'], S0); res = {'PXA': (S, b)}
    for add in ['OIK', 'HVX', 'REL', 'Z96']:
        o = [add, 'PXA'] if add != 'Z96' else ['PXA', 'Z96']
        S2, b2 = cd(D, F, o, json.loads(json.dumps(S)), rounds=1, log=False)
        ok = b2['net'] >= b['net'] * 1.06 and b2['p98'] <= b['p98'] * 1.10 and b2['pf'] >= b['pf'] - 0.05
        print('EK', o, {m: S2[m] for m in o}, {k: b2[k] for k in KS}, 'KABUL' if ok else 'RED', flush=True); res['+'.join(o)] = (S2, b2)
    json.dump({k: dict(S=v[0]) for k, v in res.items()}, open('sysB.json', 'w'))
