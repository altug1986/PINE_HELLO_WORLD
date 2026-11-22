import numpy as np, pandas as pd
from v55 import D, t, n, simR
from eng import equity, CASH
from met import SPLIT
def daily(R):
    o = R['open']; cl, mtm, worst = equity(n, R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl'], D['h'], D['l'], D['c'], o[0], o[1], o[2], o[3])
    s = pd.Series(mtm, index=pd.to_datetime(t, unit='ms')).resample('D').last().ffill(); return s.diff().fillna(s.iloc[0])
def mask_from(dp, W=7, k=3):
    roll = dp.rolling(W).sum(); M = pd.Series(False, index=dp.index)
    for y, r in roll.groupby(roll.index.year):
        r = r.copy()
        for _ in range(k):
            if r.max() <= 0 or r.isna().all(): break
            e = r.idxmax(); st = e - pd.Timedelta(days=W - 1); M[st:e] = True; r[st - pd.Timedelta(days=W - 1):e + pd.Timedelta(days=W - 1)] = np.nan
    return M
def layer_stats(dp):
    eq = dp.cumsum(); uw = eq.cummax() - eq; yrs = len(dp) / 365.25; mdd = uw.max(); w = dp.resample('W').sum()
    return dict(net=round(dp.sum()), sharpe=round(dp.mean() / dp.std() * np.sqrt(365), 2), sortino=round(dp.mean() / dp[dp < 0].std() * np.sqrt(365), 2),
                calmar=round(dp.sum() / yrs / max(mdd, 1), 2), cvW=round(w.std() / max(w.mean(), 1e-9), 2), uw=round((uw > 0).mean() * 100), p95=round(uw.quantile(.95)), p99=round(uw.quantile(.99)), mdd=round(mdd))
def teco(R, M):
    dp = daily(R); m = M.reindex(dp.index).fillna(False)
    full = layer_stats(dp); core = layer_stats(dp.where(~m, 0.0)); tail = round(dp[m].sum())
    l12 = dp[dp.index >= dp.index[-1] - pd.Timedelta(days=365)]; lm = m[l12.index]
    return full, core, tail, round(l12.sum()), round(l12[~lm].sum())
