import itertools, numpy as np, pandas as pd
from numba import njit
from p1m import build, run, report, trades, SPL, raw
D = build(5); c = D['c']; t = D['t']; R_ = raw(); oT, oV = R_['oT'], R_['oV']; f5 = t // 300000 * 300000
j = np.searchsorted(oT, f5 - 300000, side='right') - 1; oi = np.where(j >= 0, oV[np.clip(j, 0, None)], np.nan); oi[f5 - 300000 < oT[0]] = np.nan
L = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]; ch = lambda a, k: (a / L(a, k) - 1) * 100
x = R_['x']; kx = (x[:, 0] // 300000).astype(np.int64); bc = pd.Series(x[:, 4], index=kx).groupby(level=0).last().reindex(t // 300000).ffill().to_numpy()
b24 = np.nan_to_num(ch(bc, 288)); p4 = np.nan_to_num(ch(c, 48)); o4 = np.nan_to_num(ch(oi, 48)); o24 = np.nan_to_num(ch(oi, 288)); hi24 = c >= pd.Series(c).rolling(288).max().to_numpy()
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
def sig(kind, a, lag=0):
    if kind == 'BTC çöküşü': s = (b24 <= -a).astype(np.int64)
    elif kind == 'Kapitülasyon': s = ((o4 <= -a[0]) & (p4 <= -a[1])).astype(np.int64)
    else: s = -(hi24 & (o24 >= a)).astype(np.int64)
    return np.r_[np.zeros(lag, np.int64), s[:-lag]] if lag else s
def fast(out, lab, **kw):
    R, _ = trades(D, out); p = R['pnl']; et = t[R['eb']]; xt = t[R['xb']]; e = t[-1]
    return dict(t=lab, net=round(p.sum()), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()), L12=round(p[xt >= e - 365 * 86400000].sum()), L200=round(p[xt >= e - 200 * 86400000].sum()), **kw)
A = {'BTC çöküşü': (6, 8, 10), 'Kapitülasyon': ((4, 3), (5, 4), (6, 5)), 'Tepede long yığılması': (3, 5, 8)}
if __name__ == '__main__':
    b = fast(run(D), 'V6.6'); print(b); rows = []
    for kind, aa in A.items():
        for a, H, w in itertools.product(aa, (144, 288, 576), (2, 4, 8)):
            rows.append(fast(run(D, xP=ent(sig(kind, a), c, H, 10.0), xw=float(w)), kind, a=str(a), H=H, w=w))
    r = pd.DataFrame(rows); r.to_pickle('edge2.pkl'); pd.set_option('display.width', 260)
    print(r.groupby('t').agg(ort=('net', 'mean'), max=('net', 'max'), baz_ustu=('net', lambda v: int((v > b['net']).sum())), y6=('net', lambda v: int((v > b['net'] * 1.06).sum())), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean')).round(0).to_string())
    for kind in A: print(r[r.t == kind].pivot_table(index=['a', 'H'], columns='w', values='net').round(0).to_string())
