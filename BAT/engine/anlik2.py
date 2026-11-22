import itertools, pickle, numpy as np, pandas as pd
from p1m import build, run, report
from eng import rma, trr
D = build(5); c = D['c']; t = D['t']; tr = trr(D['h'], D['l'], c, True); atr = rma(tr, 14)
x5 = np.nan_to_num(tr / np.r_[np.nan, atr[:-1]])                                                     # kapanan 5 dk mumun TR / önceki ATR14
m1 = pickle.load(open('D1.pkl', 'rb')); r1 = np.abs(np.r_[0, m1['c'][1:] / m1['c'][:-1] - 1]); a1 = pd.Series(r1).rolling(60).mean().to_numpy()
j = np.searchsorted(m1['t'], t + 300000) - 1; x1 = np.nan_to_num(r1[j] / a1[j])                       # kapanan mumun son 1 dk hareketi / 60 dk ort.
rows = [report(D, run(D), 'V6.6 baz')]
for (src, X, ks), f, xa in itertools.product([('5 dk TR/ATR', x5, (2, 3, 5)), ('son 1 dk / 60 dk ort.', x1, (4, 6, 10))], (0.9, 0.75), (0.0, 0.02, 0.05)):
    for k in ks:
        hi = X >= k; rows.append(dict(report(D, run(D, hi=hi, hfac=f, xadd=xa), f'{src} ≥ {k} · giriş ×{f} · çıkış +{xa}'), olay=round(hi.mean() * 100, 2)))
r = pd.DataFrame(rows); pd.set_option('display.width', 250); b = r.iloc[0]
print(r.iloc[:1].to_string(index=False)); print('baz üstü:', int((r.net > b.net).sum()), '/', len(r) - 1, '| %6 üstü:', int((r.net > b.net * 1.06).sum()))
print(r.iloc[1:].sort_values('IS', ascending=False).head(8).to_string(index=False))
print(r.iloc[1:].sort_values('IS').head(3).to_string(index=False))
