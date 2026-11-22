import numpy as np, pandas as pd
import v55
from v55 import *
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
tg, why = run(A); R = simR(D, tg); rows = []
for eb, xb, s, ep, q, pn in zip(R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl']):
    pl = (c[eb + 1:xb + 1] / ep - 1) * s * 100
    for L in (4.0, 5.0):
        k = np.nonzero(pl <= -L)[0]
        if len(k):
            i = eb + 1 + k[0]; cut = (c[i] - TICK * s - ep) * q * s - 0.0004 * q * (ep + c[i])
            rows.append(dict(L=L, i=i, dr=s, pnl=pn, cut=cut, gain=cut - pn, held=i - eb, IS=t[xb] < SPLIT, why=why[xb],
                             a15=A15[i], d15=D15[i] * s, a60=TRF['a60'][0][i], d60=TRF['a60'][1][i] * s, a240=TRF['a240'][0][i], d240=TRF['a240'][1][i] * s,
                             x60=BADX[i], dx60=BDI[i] * s, x240=TRF['x240'][0][i], dx240=TRF['x240'][1][i] * s, bv=BV[i], nr=NR[i], m12=MOM[12][i] * s, m48=MOM[48][i] * s, vr=VR[i]))
X = pd.DataFrame(rows); pd.set_option('display.width', 250)
for L in (4.0, 5.0):
    x = X[X.L == L]; print(f"\n=== −%{L:.0f}'e değen işlem: {len(x)} | sonunda kârlı kapanan %{(x.pnl > 0).mean()*100:.0f} | şimdiki sonuç {x.pnl.sum():.0f}$ | orada kesilseydi {x.cut.sum():.0f}$ | kesmenin etkisi {x.gain.sum():+.0f}$ (IS {x.gain[x.IS].sum():+.0f} / OOS {x.gain[~x.IS].sum():+.0f})")
    for f in ['a15', 'd15', 'a60', 'd60', 'a240', 'd240', 'x60', 'dx60', 'x240', 'dx240', 'bv', 'nr', 'm12', 'm48', 'vr', 'held']:
        b = pd.qcut(x[f], 5, labels=False, duplicates='drop'); g = x.groupby([b, x.IS]).gain.sum().unstack().fillna(0)
        print(f"{f:6s} kesme etkisi Q1..Q5  IS:", ' '.join(f"{v:+6.0f}" for v in g[True]), '| OOS:', ' '.join(f"{v:+6.0f}" for v in g[False]), '| eşikler', np.round(x[f].quantile([.2, .4, .6, .8]).values, 1))
X.to_pickle('cut55.pkl')
