import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01, z03
D = ztf.build(1, thr=25.0); B = D['B']; c, h, l, o, a = D['c'], D['h'], D['l'], D['o'], D['atr']; n = D['n']; mid = B.index[D['mid']]
sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]; r10 = (c / sh(c, 10) - 1) * 100; z = np.zeros(n, np.bool_)
aL = (D['L'] >= 3).astype(np.bool_); aS = ((D['Sh'] >= 3) & (D['L'] < 3)).astype(np.bool_); spk = np.nan_to_num(r10 >= 3).astype(np.bool_)
rows = []
for X, K, keep in [(99, 0, 0)] + list(itertools.product([1.5, 2, 3, 5], [2, 4, 8], [0.0, 0.5, 0.7])):
    up = np.nan_to_num(r10 >= X).astype(np.bool_); dn = np.nan_to_num(r10 <= -X).astype(np.bool_)
    eq, T = z03.sim8(o, h, l, c, a, aL, aS, 11.0, 960, spk, 60, up, dn, float(K), float(keep))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'guard', 'mfe', 'lk']); t['et'] = B.index[t.ei.astype(int)]
    e = pd.Series(eq, index=B.index); f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None)
    k_ = t[t.lk == 1]; give = ((k_.mfe * 100) - (k_.pnl / 5)).mean() if len(k_) else np.nan       # kilitli işlemlerde geri verilen (puan %)
    allgive = ((t.mfe * 100) - (t.pnl / 5)).mean()
    rows.append(dict(X=X, K=K, keep=keep, n=f['n'], net=f['net'], IS=A['net'], OOS=Bm['net'], PF=f['PF'], MaxDD=f['MaxDD'], P99=f['P99'], Cal=f['Calmar'], s10=f['s10'], s1h=f['s1h'],
                     nKilit=len(k_), kilitPnl=k_.pnl.sum(), geriVerKilit=give, geriVerTüm=allgive, maeMax=f['maeMax']))
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/LOCK_1DK.csv', index=False)
pd.set_option('display.width', 250); print(R.round(2).to_string(index=False))
