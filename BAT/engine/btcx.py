import itertools, numpy as np, pandas as pd
from numba import njit
from edge2 import D, c, t, run, report, trades, ent, sig, b24, bc
from p1m import SPL
cap = ent(sig('Kapitülasyon', (5, 4)), c, 144, 5.0); n = len(c); yrs = (t[-1] - t[0]) / 3.15576e10
L = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]; b1 = np.nan_to_num((bc / L(bc, 12) - 1) * 100); a1 = np.nan_to_num((c / L(c, 12) - 1) * 100)
@njit(cache=True)
def ent2(s, c, hold, sl):                      # stop sonrası sinyal sönmeden yeniden girme
    m = len(s); out = np.zeros(m, np.int64); v = 0; e = 0.0; vb = 0; blk = 0
    for i in range(m):
        if v != 0 and ((c[i] / e - 1) * v <= -sl / 100 or i - vb >= hold):
            if (c[i] / e - 1) * v <= -sl / 100: blk = 1
            v = 0
        if s[i] == 0: blk = 0
        if s[i] != 0 and blk == 0:
            if v != s[i]: v = s[i]; e = c[i]; vb = i
        out[i] = v
    return out
def ev(ov, lab, **kw):
    R, op = trades(D, run(D, ovr=ov)); cum = np.zeros(n); np.add.at(cum, R['xb'], R['pnl']); cum = np.cumsum(cum); opn = np.zeros(n); wo = np.zeros(n)
    for eb, xb, d, ep, q in zip(R['eb'], R['xb'], R['dir'], R['ep'], R['q']): s = slice(eb + 1, xb); opn[s] = (c[s] - ep) * q * d; wo[s] = ((D['l'][s] if d > 0 else D['h'][s]) - ep) * q * d
    ix = pd.to_datetime(t + 300000, unit='ms'); E = pd.Series(cum + opn, index=ix).resample('30min').last().ffill(); Wn = pd.Series(cum + wo, index=ix).resample('30min').min().reindex(E.index).fillna(E)
    pk = E.cummax(); uw = pk - E; uwW = pk - Wn; mo = E.resample('ME').last().diff().dropna(); net = R['pnl'].sum(); r2 = np.corrcoef(np.arange(len(E)), E.to_numpy())[0, 1] ** 2; cv = mo.std() / mo.mean()
    p = R['pnl']; et = t[R['eb']]
    return dict(t=lab, net=round(net), n=len(p), PF=round(p[p > 0].sum() / -p[p < 0].sum(), 3), IS=round(p[et < SPL].sum()), OOS=round(p[et >= SPL].sum()), MaxDD=round(uwW.max()), MaxDDt=str(uwW.idxmax().date()),
                Cal=round(net / yrs / uw.max(), 2), P=f"{uw.quantile(.95):.0f}/{uw.quantile(.98):.0f}/{uw.quantile(.99):.0f}", BoB=round((net / yrs / uw.quantile(.98)) * r2 ** 2 * (mo > 0).mean() / (1 + 0.25 * min(cv, 8)), 2), **kw)
if __name__ == '__main__':
    rows = [ev(cap, 'V6.8'), ev(np.where(cap != 0, cap, ent(sig('BTC çöküşü', 10), c, 288, 10.0)), 'V6.8 + BTC çöküşü (eski)')]
    print(rows[0], '\n', rows[1], flush=True)
    for thr, H, sl, cf, re in itertools.product((8, 10, 12), (144, 288), (3.0, 5.0, 10.0), ('yok', 'BTC 1s ≥ 0', 'BAT 1s > 0'), (0, 1)):
        s = (b24 <= -thr) & ({'yok': True, 'BTC 1s ≥ 0': b1 >= 0, 'BAT 1s > 0': a1 > 0}[cf]); s = s.astype(np.int64)
        b = ent2(s, c, H, sl) if re else ent(s, c, H, sl)
        rows.append(ev(np.where(cap != 0, cap, b), 'V6.8 + BTC', thr=thr, H=H, sl=sl, teyit=cf, stopsonra=('bekle' if re else 'gir')))
    r = pd.DataFrame(rows); r.to_pickle('btcx.pkl'); pd.set_option('display.width', 260)
    ok = r.iloc[2:][(r.iloc[2:].MaxDD <= 300)]; print('MaxDD ≤ 300 olan:', len(ok), '/', len(r) - 2)
    print(ok.sort_values('BoB', ascending=False).head(12).to_string(index=False))
