import numpy as np, pandas as pd
from tv54 import tvload
from v55 import run, D, simR, t, n
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
T = tvload('tv/v55_tv.csv'); cp = np.r_[0, T.pnl.cumsum()]
print('TV V5.5:', len(T), 'işlem | net', round(T.pnl.sum()), '| PF', round(T.pnl[T.pnl > 0].sum() / -T.pnl[T.pnl < 0].sum(), 3), '| MaxDD kapanış', round((np.maximum.accumulate(cp) - cp).max()))
tg, why = run(A); R = simR(D, tg); M = pd.DataFrame(dict(et=t[R['eb']], xt=t[R['xb']], dr=R['dir'], pnl=R['pnl']))
a, b = T.et.min(), min(T.xt.max(), t[-1]); Mm = M[(M.et >= a) & (M.xt <= b)]; Tm = T[T.xt <= b]
both = set(zip(Tm.et, Tm.dr)) & set(zip(Mm.et, Mm.dr))
print('aynı dönem TV', len(Tm), round(Tm.pnl.sum()), '| motor', len(Mm), round(Mm.pnl.sum()), f'| eşleşme %{len(both)/len(Tm)*100:.1f}')
L12 = T.xt >= T.xt.max() - 365 * 86400000; print('TV son 12 ay:', round(T.pnl[L12].sum()), '| long', round(T.pnl[L12 & (T.dr > 0)].sum()), 'short', round(T.pnl[L12 & (T.dr < 0)].sum()))
x = pd.to_datetime(T.xt, unit='ms'); y = T.groupby(x.dt.year).pnl.sum().round(0); print('TV yıl $:', y.to_dict())
T.to_pickle('tv/v55_tv.pkl')
