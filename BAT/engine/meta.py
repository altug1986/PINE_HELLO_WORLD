import numpy as np, pandas as pd, itertools
from sklearn.tree import DecisionTreeClassifier, export_text
from sysB import *
from met import metrics, SPLIT
# META-ETİKET: birincil model V1.0 (24s dönüş). İkincil karar ağacı her işlem girişinde "Binance ekranı" özelliklerine bakıp
# AL (dönüşü oyna) / ATLA (flat) / ÇEVİR (hareketi takip et: aynı süre ters yön) der.
D = load(5); F = feats(D); c = D['c']; h = D['h']; l = D['l']; v = pd.Series(D['v']); t = D['t']; n = len(c)
R0 = simR(D, mot2(D, F, dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)))
o = np.r_[np.nan, D['oi'][:-1]]; xc = D['xc']
def sh(a, k): return np.r_[np.full(k, np.nan), a[:-k]]
hi24 = pd.Series(h).rolling(288).max().to_numpy(); lo24 = pd.Series(l).rolling(288).min().to_numpy()
FE = dict(
    güç=np.abs(c / sh(c, 288) - 1) * 100,                                   # 24s değişim büyüklüğü
    aralık=(c - lo24) / np.maximum(hi24 - lo24, 1e-12),                     # 24s aralık içindeki yer (0 dip, 1 tepe)
    hacim24=(v.rolling(288).sum() / v.rolling(288).sum().shift(288)).to_numpy(),
    oi24=(o / sh(o, 288) - 1) * 100, oi1s=(o / sh(o, 12) - 1) * 100,
    btc24=(xc / sh(xc, 288) - 1) * 100, bat1s=(c / sh(c, 12) - 1) * 100, bat7g=(c / sh(c, 2016) - 1) * 100,
    atr1s=D['atr1h'], saat=((t // 3600000) % 24).astype(float))
eb, xb, dr, pn = R0['eb'], R0['xb'], R0['dir'], R0['pnl']
X = pd.DataFrame({k: v_[eb] for k, v_ in FE.items()})
for k in ['oi24', 'oi1s', 'btc24', 'bat1s', 'bat7g']: X[k] = X[k] * dr     # yön işaretli: + = işlem yönünde
X['aralık'] = np.where(dr > 0, X['aralık'], 1 - X['aralık'])              # 0 = işlem yönünün en kötü ucunda
y = (pn > 0).astype(int); IS = t[xb] < SPLIT; X = X.fillna(X[IS].median())
q = R0['q']; flip = -pn - 2 * 0.0004 * q * (R0['ep'] + R0['xp']) - 4 * TICK * q     # aynı süre ters yön (yaklaşık maliyet)
rows = []
for depth, leaf in itertools.product([1, 2, 3], [40, 60, 100, 150]):
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=0).fit(X[IS], y[IS])
    pr = m.predict_proba(X)[:, 1]; base = y[IS].mean()
    for lo, mode in itertools.product([0.40, 0.45, 0.50], ['ATLA', 'ÇEVİR']):
        bad = pr < lo; alt = np.where(bad, 0 if mode == 'ATLA' else flip, pn)
        rows.append(dict(derinlik=depth, yaprak=leaf, eşik=lo, mod=mode, kötü_IS=int(bad[IS].sum()), kötü_OOS=int(bad[~IS].sum()),
                         IS=round(alt[IS].sum()), OOS=round(alt[~IS].sum()), dIS=round(alt[IS].sum() - pn[IS].sum()), dOOS=round(alt[~IS].sum() - pn[~IS].sum())))
df = pd.DataFrame(rows); pd.set_option('display.width', 250)
print('BAZ V1.0 işlem: IS', round(pn[IS].sum()), 'OOS', round(pn[~IS].sum()), '| IS kazanma', round(y[IS].mean(), 3))
print('IS iyileştiren', (df.dIS > 0).mean().round(2), '| aynı ayarlarda OOS iyileştiren', (df[df.dIS > 0].dOOS > 0).mean().round(2))
print(df.sort_values('dIS', ascending=False).head(10).to_string(index=False))
best = df.sort_values('dIS', ascending=False).iloc[0]
m = DecisionTreeClassifier(max_depth=int(best.derinlik), min_samples_leaf=int(best.yaprak), random_state=0).fit(X[IS], y[IS])
print(export_text(m, feature_names=list(X.columns), decimals=3))
pd.to_pickle(dict(X=X, y=y, IS=IS, pn=pn, flip=flip, eb=eb, xb=xb, dr=dr), 'meta.pkl')
