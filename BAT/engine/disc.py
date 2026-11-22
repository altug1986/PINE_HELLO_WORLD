import numpy as np, pandas as pd
from v55 import *
z = np.load('d5.npz'); oi = z['oi'].astype(float); oi[oi <= 0] = np.nan; oi = pd.Series(oi).ffill().to_numpy()
tg, why = run({}); R = simR(D, tg); X = pd.DataFrame(dict(eb=R['eb'], xb=R['xb'], dr=R['dir'], pnl=R['pnl'])); X['why'] = why[X.xb]; X['IS'] = t[X.xb] < SPLIT
i = X.eb.to_numpy(); dr = X.dr.to_numpy()
vq = D['v'] * c; VQ = pd.Series(vq).rolling(6).sum(); VQ = (VQ / VQ.rolling(1728).mean()).to_numpy()
hh = pd.Series(c).rolling(2016).max().to_numpy(); ll = pd.Series(c).rolling(2016).min().to_numpy(); pos7 = (c - ll) / (hh - ll)
x4 = z['x240']; from eng import dmi, htf_map; p4, m4, a4 = dmi(x4[1], x4[2], x4[3], 14, 14); A4 = htf_map(a4, z['x240_k'], z['x240_map']); DI4 = htf_map(p4 - m4, z['x240_k'], z['x240_map'])
doi = np.r_[np.full(288, np.nan), (oi[288:] / oi[:-288] - 1) * 100]
lastS = np.full(len(X), 99999); ls = {1: -10**9, -1: -10**9}
for k in range(len(X)):
    lastS[k] = i[k] - ls[dr[k]]
    if X.why.iat[k] == 2: ls[dr[k]] = X.xb.iat[k]
G = dict(d288A=F['d288'][i] * dr, m864A=F['m864'][i] * dr, m2016A=F['m2016'][i] * dr, di15A=D15[i] * dr, adx15=A15[i], badx=BADX[i], bdiA=BDI[i] * dr, nr=NR[i], bv=BV[i], vq=VQ[i], volr=VR[i],
         b288A=F['b288'][i] * dr, pos7A=np.where(dr > 0, pos7[i], 1 - pos7[i]), adx4=A4[i], di4A=DI4[i] * dr, doi=doi[i], sinceStop=np.minimum(lastS, 5000))
X['st'] = X.why == 2
for k, v in G.items():
    X[k] = v; ok = X[k].notna(); b = pd.qcut(X.loc[ok, k], 5, labels=False, duplicates='drop')
    g = X[ok].groupby([b, X.IS[ok]]).agg(st=('st', 'mean'), net=('pnl', 'sum')).unstack()
    print(f"{k:9s} stop% IS:", ' '.join(f"{v*100:4.0f}" for v in g[('st', True)]), '| OOS:', ' '.join(f"{v*100:4.0f}" for v in g[('st', False)]),
          '|| net IS:', ' '.join(f"{v:6.0f}" for v in g[('net', True)]), '| OOS:', ' '.join(f"{v:6.0f}" for v in g[('net', False)]))
X.to_pickle('disc.pkl')
