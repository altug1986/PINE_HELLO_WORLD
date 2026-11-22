import numpy as np, numba as nb, pandas as pd
from eng import equity, TICK, CASH, COMM
SPLIT = pd.Timestamp('2024-11-01').value // 10**6  # IS < SPLIT <= OOS (exit time)

def metrics(D, R, lab=''):
    t = D['t']; n = len(t)
    o = R['open']
    cl, mtm, worst = equity(n, R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl'], D['h'], D['l'], D['c'], o[0], o[1], o[2], o[3])
    pnl = R['pnl']; xt = t[R['xb']]
    gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    day = pd.to_datetime(t, unit='ms').normalize()
    s = pd.Series(mtm, index=day).groupby(level=0).last()
    dp = s.diff().fillna(s.iloc[0]); r = dp / CASH
    yrs = (t[-1] - t[0]) / 3.15576e10
    uw = s.cummax() - s
    clp = np.r_[0, np.cumsum(pnl)]; mdd_c = (np.maximum.accumulate(clp) - clp).max()
    mdd_i = (np.maximum.accumulate(mtm) - worst).max()
    net = pnl.sum() + (mtm[-1] - cl[-1])
    ulcer = np.sqrt((uw ** 2).mean())
    mo = s.resample('ME').last().diff().fillna(s.resample('ME').last().iloc[0])
    yr = pd.Series(pnl, index=pd.to_datetime(xt, unit='ms')).groupby(lambda x: x.year).sum() / CASH * 100
    last12 = pnl[xt >= t[-1] - 365 * 86400000].sum()
    fee = (COMM * R['q'] * (R['ep'] + R['xp'])).sum()
    return dict(lab=lab, net=round(net), pct=round(net / CASH * 100), n=len(pnl), pf=round(gp / max(gl, 1e-9), 3),
                win=round((pnl > 0).mean() * 100, 1) if len(pnl) else 0,
                sharpe=round(r.mean() / r.std() * np.sqrt(365), 2), sortino=round(r.mean() / r[r < 0].std() * np.sqrt(365), 2),
                calmar=round(net / yrs / max(mdd_c, 1), 2), p95=round(uw.quantile(.95)), p98=round(uw.quantile(.98)),
                p99=round(uw.quantile(.99)), mddc=round(mdd_c), mddi=round(mdd_i), ulcer=round(ulcer, 1),
                upi=round(net / yrs / max(ulcer, 1e-9), 1), IS=round(pnl[xt < SPLIT].sum()), OOS=round(pnl[xt >= SPLIT].sum()),
                posM=round((mo > 0).mean() * 100), last12=round(last12), fee=round(fee),
                yr=' '.join(f"{k%100}:{v:.0f}" for k, v in yr.items()))

# ---------------- monkey ----------------
@nb.njit(cache=True)
def tpnl(c, s, d, dr):
    e = c[s] + TICK * dr; x = c[s + d] - TICK * dr; q = CASH / c[s]
    return (x - e) * q * dr - COMM * q * (e + x)

@nb.njit(cache=True)
def mk_dir(c, eb, xb, N, seed):
    np.random.seed(seed); out = np.zeros(N)
    for r in range(N):
        s = 0.0
        for k in range(len(eb)):
            s += tpnl(c, eb[k], xb[k] - eb[k], 1 if np.random.random() < .5 else -1)
        out[r] = s
    return out

@nb.njit(cache=True)
def mk_reg(c, eb, xb, dr, key, order, start, cnt, N, seed):
    # relocate each trade to a random bar with the same regime key (key=-1 => anywhere). same duration+direction
    np.random.seed(seed); out = np.zeros(N); n = len(c)
    for r in range(N):
        s = 0.0
        for k in range(len(eb)):
            d = xb[k] - eb[k]; kk = key[eb[k]]
            for _ in range(50):
                j = order[start[kk] + np.random.randint(cnt[kk])]
                if j + d < n: break
            if j + d >= n: j = n - d - 1
            s += tpnl(c, j, d, dr[k])
        out[r] = s
    return out

def prep_key(key):
    key = key.astype(np.int64); key[key < 0] = key.max() + 1
    order = np.argsort(key, kind='stable'); u, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    m = np.zeros(key.max() + 1, np.int64); st = np.zeros_like(m); ct = np.ones_like(m)
    st[u] = start; ct[u] = cnt
    return key, order, st, ct

def terc(x, valid):
    q = np.nanquantile(x[valid], [1/3, 2/3]); k = np.digitize(x, q); k[~valid] = -1; return k

def regimes(D, warm=2000):
    n = len(D['c']); ok = np.arange(n) >= warm
    a = D['adx30']; v = D['atr1h']
    k1 = terc(a, ok & ~np.isnan(a)); k2 = terc(v, ok & ~np.isnan(v))
    bf = np.where((D['badx4'] > 30) & ~np.isnan(D['bdi4']), np.sign(D['bdi4']), 0).astype(int) + 1
    sma = pd.Series(D['c']).rolling(14400, min_periods=1).mean().to_numpy()  # ~50 gün
    k4 = (D['c'] > sma).astype(int)
    q = pd.to_datetime(D['t'], unit='ms'); k6 = ((q.year - 2020) * 4 + (q.month - 1) // 3).to_numpy()
    k5 = np.where((k1 >= 0) & (k2 >= 0), k1 * 9 + k2 * 3 + bf, -1)
    anyk = np.where(ok, 0, -1)
    fam = dict(ZAMAN=anyk, ADX=k1, VOL=k2, BTC=np.where(ok, bf, -1), TREND50G=np.where(ok, k4, -1), KOMBO=k5, CEYREK=np.where(ok, k6, -1))
    return {k: prep_key(v) for k, v in fam.items()}

def monkey(D, R, RG, N=1000, seed=7):
    c = D['c']; eb, xb, dr = R['eb'], R['xb'], R['dir']; net = R['pnl'].sum()
    res = {}
    m = mk_dir(c, eb, xb, N, seed); res['YÖN'] = (m > net).mean()
    for f, (key, order, st, ct) in RG.items():
        m = mk_reg(c, eb, xb, dr, key, order, st, ct, N, seed); res[f] = (m > net).mean()
    return res
