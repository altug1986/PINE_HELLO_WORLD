import numpy as np, pandas as pd
from zz import g_rev
from eng import load, TICK, CASH, COMM
from lib import simR
from met import SPLIT
# 3/3 takipçinin her işlemi için giriş anındaki bağlam (yön işaretli) + takip / ters K/Z
D = load(5); c = D['c']; t = D['t']; n = len(c)
def sh(a, k): return np.r_[np.full(k, np.nan), a[:-k]]
o = np.r_[np.nan, D['oi'][:-1]]; xc = D['xc']; v = pd.Series(D['v'])
atr = D['atr1h']; atrMed = pd.Series(atr).rolling(8640, min_periods=288).median().to_numpy()
hi24 = pd.Series(D['h']).rolling(288).max().to_numpy(); lo24 = pd.Series(D['l']).rolling(288).min().to_numpy()
BAR = dict(e24=(c / sh(c, 288) - 1) * 100, e4s=(c / sh(c, 48) - 1) * 100, e7g=(c / sh(c, 2016) - 1) * 100,
           btc24=(xc / sh(xc, 288) - 1) * 100, oi24=(o / sh(o, 288) - 1) * 100, oi4s=(o / sh(o, 48) - 1) * 100,
           volr=atr / atrMed, hac24=(v.rolling(288).sum() / v.rolling(288).sum().shift(288)).to_numpy(),
           aralık=(c - lo24) / np.maximum(hi24 - lo24, 1e-12), saat=((t // 3600000) % 24).astype(float), gün=((t // 86400000 + 3) % 7).astype(float))
SIGNED = ['e24', 'e4s', 'e7g', 'btc24', 'oi4s']
def trades(u=3, d=3):
    s = g_rev(c, u, d); R = simR(D, s); eb, xb, dr = R['eb'], R['xb'], R['dir']
    X = pd.DataFrame({k: a[eb] for k, a in BAR.items()})
    for k in ['e24', 'e4s', 'e7g', 'btc24']: X[k] = X[k] * dr
    X['aralık'] = np.where(dr > 0, X['aralık'], 1 - X['aralık'])
    q = R['q']; X['yön'] = dr; X['pnl'] = R['pnl']; X['ters'] = -R['pnl'] - 2 * COMM * q * (R['ep'] + R['xp']) - 4 * TICK * q
    X['eb'] = eb; X['xb'] = xb; X['xt'] = t[xb]
    return X, s
if __name__ == '__main__':
    X, s = trades(); IS = X.xt < SPLIT; mid = np.quantile(X.xt[IS], 0.5)
    print('işlem', len(X), '| takip net', round(X.pnl.sum()), '| ters net', round(X.ters.sum()), '| brüt kâr havuzu', round(X.pnl[X.pnl > 0].sum()))
    rows = []
    for f in ['e24', 'e4s', 'e7g', 'btc24', 'oi24', 'oi4s', 'volr', 'hac24', 'aralık', 'saat', 'gün']:
        x = X[f]; qs = np.nanquantile(x[IS], [0.2, 0.4, 0.6, 0.8]) if f not in ('saat', 'gün') else ([4, 8, 12, 16, 20] if f == 'saat' else [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        b = np.digitize(x, qs)
        for yön in [1, -1]:
            for k in np.unique(b):
                m = (b == k) & (X.yön == yön); a = m & IS & (X.xt < mid); bb = m & IS & (X.xt >= mid); oo = m & ~IS
                rows.append(dict(özellik=f, yön='L' if yön > 0 else 'S', kova=k, IS_a=X.pnl[a].mean(), IS_b=X.pnl[bb].mean(), OOS=X.pnl[oo].mean(), n_IS=int((m & IS).sum())))
    df = pd.DataFrame(rows)
    # kalıcılık: IS iki yarısında kova ortalamalarının korelasyonu (yön × kova)
    out = df.groupby('özellik').apply(lambda g: pd.Series(dict(korelasyon_ISa_ISb=round(g.IS_a.corr(g.IS_b), 2), korelasyon_IS_OOS=round(((g.IS_a + g.IS_b) / 2).corr(g.OOS), 2),
                                                             yayılım_IS=round(((g.IS_a + g.IS_b) / 2).max() - ((g.IS_a + g.IS_b) / 2).min(), 2))), include_groups=False)
    print(out.sort_values('korelasyon_ISa_ISb', ascending=False).to_string())
    pd.set_option('display.width', 200); print(df[df.özellik == 'e24'].round(2).to_string(index=False))
    X.to_pickle('ctx_trades.pkl')
