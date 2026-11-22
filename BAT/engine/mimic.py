import numpy as np, pandas as pd
from short_top_lib import DIVh, zh, to5, D, t, n, c
from met import SPLIT
z5 = np.load('d5.npz'); oi = z5['oi'].astype(float); oi[oi <= 0] = np.nan; oi = pd.Series(oi).ffill().to_numpy()
ix = pd.to_datetime(t, unit='ms'); o = D['o']; v = D['v']; xc = z5['xc']
df = pd.DataFrame(dict(c=c, oi=oi, v=v, sv=v * np.sign(c - o), xc=xc), index=ix)
H = df.resample('1h', label='left').agg(dict(c='last', oi='last', v='sum', sv='sum', xc='last'))
F = pd.DataFrame(index=H.index); lc, lo, lx = np.log(H.c), np.log(H.oi), np.log(H.xc)
for k in (1, 4, 24, 72, 168, 720):
    F[f'r{k}'] = lc - lc.shift(k); F[f'oi{k}'] = lo - lo.shift(k); F[f'bx{k}'] = lx - lx.shift(k)
    F[f'vd{k}'] = H.sv.rolling(k).sum() / H.v.rolling(k).sum()
    F[f'lb{k}'] = F[f'oi{k}'] * np.sign(F[f'r{k}'])                 # OI artışı fiyat yönünde mi (long/short birikimi)
for k in (24, 168, 720):
    F[f'oiz{k}'] = (lo - lo.rolling(k).mean()) / lo.rolling(k).std(); F[f'pz{k}'] = (lc - lc.rolling(k).mean()) / lc.rolling(k).std()
    F[f'vz{k}'] = (np.log(H.v + 1) - np.log(H.v + 1).rolling(k).mean()) / np.log(H.v + 1).rolling(k).std()
Y = pd.DataFrame(dict(z720=zh(DIVh, 720), z24=zh(DIVh, 24)), index=DIVh.index).reindex(F.index)
X = F.replace([np.inf, -np.inf], np.nan); ok = X.notna().all(1) & Y.z720.notna() & Y.z24.notna()
sp = pd.Timestamp(SPLIT, unit='ms'); IS = ok & (X.index < sp); OS = ok & (X.index >= sp)
print('saat: IS', IS.sum(), 'OOS', OS.sum())
cor = pd.DataFrame({y: {f: (np.corrcoef(X[f][IS], Y[y][IS])[0, 1], np.corrcoef(X[f][OS], Y[y][OS])[0, 1]) for f in X} for y in ['z720', 'z24']})
best = sorted(X.columns, key=lambda f: -abs(cor['z720'][f][0]))[:8]; print('z720 en güçlü korelasyon (IS / OOS):', {f: tuple(round(x, 2) for x in cor['z720'][f]) for f in best})
best24 = sorted(X.columns, key=lambda f: -abs(cor['z24'][f][0]))[:8]; print('z24 en güçlü korelasyon (IS / OOS):', {f: tuple(round(x, 2) for x in cor['z24'][f]) for f in best24})
from numpy.linalg import lstsq
for y in ['z720', 'z24']:
    A = np.c_[np.ones(IS.sum()), X[IS].values]; m = X[IS].mean(); s = X[IS].std()
    Ai = np.c_[np.ones(IS.sum()), ((X[IS] - m) / s).values]; Ao = np.c_[np.ones(OS.sum()), ((X[OS] - m) / s).values]
    lam = 50.0; w = np.linalg.solve(Ai.T @ Ai + lam * np.eye(Ai.shape[1]), Ai.T @ Y[y][IS].values)
    pi, po = Ai @ w, Ao @ w; print(f'{y} doğrusal taklit: IS korelasyon {np.corrcoef(pi, Y[y][IS])[0,1]:.2f} | OOS korelasyon {np.corrcoef(po, Y[y][OS])[0,1]:.2f}')
    Pz = pd.Series(np.nan, index=X.index); Pz[ok] = np.c_[np.ones(ok.sum()), ((X[ok] - m) / s).values] @ w; Pz.to_pickle(f'mimic_{y}.pkl')
X.to_pickle('mimic_X.pkl')
