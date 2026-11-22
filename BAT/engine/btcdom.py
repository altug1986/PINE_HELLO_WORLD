import itertools, numpy as np, pandas as pd
from numba import njit
from p1m import build, run, report, trades, SPL
D = build(5); c = D['c']; t = D['t']
bd = pd.read_csv('btcdom1m.csv.gz', usecols=['time', 'close']); dom = bd.groupby(bd.time // 300000).close.last().reindex(t // 300000).ffill().to_numpy().copy()
dom[t < bd.time.iloc[0]] = np.nan
@njit(cache=True)
def ent(s, c, hold, sl):
    m = len(s); out = np.zeros(m, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(m):
        if v != 0 and ((c[i] / e - 1) * v <= -sl / 100 or i - vb >= hold): v = 0
        if s[i] != 0:
            if v != s[i]: v = s[i]; e = c[i]; vb = i
            else: vb = i
        out[i] = v
    return out
def chg(N, lag=0):
    x = (dom / np.r_[np.full(N, np.nan), dom[:-N]] - 1) * 100
    return np.r_[np.full(lag, np.nan), x[:-lag]] if lag else x
def fast(out, lab, **kw):
    R, _ = trades(D, out); p = R['pnl']; et = t[R['eb']]; xt = t[R['xb']]; e = t[-1]
    return dict(t=lab, net=round(p.sum()), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()),
                L12=round(p[xt >= e - 365 * 86400000].sum()), L200=round(p[xt >= e - 200 * 86400000].sum()), **kw)
def variants(N, qq, lag=0):
    x = chg(N, lag); thr = np.nanquantile(np.abs(chg(N)[t < SPL]), qq); xn = np.nan_to_num(x)
    yield 'kapı', dict(xGL=xn <= -thr, xGS=xn >= thr), thr
    for H, w in itertools.product((144, 288), (1.0, 2.0, 4.0)):
        s = np.where(xn >= thr, 1, np.where(xn <= -thr, -1, 0)).astype(np.int64); yield f'oy w{w:g} tut{H}', dict(xP=ent(s, c, H, 10.0), xw=w), thr
if __name__ == '__main__':
    b = fast(run(D), 'V6.6'); rows = []
    for N, qq in itertools.product((48, 288, 2016), (0.6, 0.75, 0.9)):
        for lab, kw, thr in variants(N, qq): rows.append(fast(run(D, **kw), lab, N=N, q=qq, esik=round(thr, 2)))
    r = pd.DataFrame(rows); r.to_pickle('btcdom.pkl'); pd.set_option('display.width', 250); print(b)
    print('baz üstü', int((r.net > b['net']).sum()), '/', len(r), '| %6 üstü', int((r.net > b['net'] * 1.06).sum()))
    print(r.groupby(['t', 'N']).net.agg(['mean', 'max']).round(0).unstack().to_string())
    print(r.sort_values('IS', ascending=False).head(10).to_string(index=False))
