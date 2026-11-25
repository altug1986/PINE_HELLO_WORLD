"""Zaman dilimi regülasyonu: R2'yi (rejim oyu + IS kural seçimi) herhangi bir TF'de AYNI SÜRELERLE kurar.
s = 2 / tf (bar sayıları s ile ölçeklenir), r = sqrt(s) (ATR eşikleri: ATR ~ sqrt(TF)). tf=2 → R2 birebir (regresyon).
"""
import numpy as np, pandas as pd
from zec_data import load
import z01, z02, regime
from path_quality import path


def rma(x, n):
    return z01.rma(x, n)


def htf(B, tf, rule, fn):
    H = B[['open', 'high', 'low', 'close']].resample(rule, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    v = pd.Series(fn(H), index=H.index + pd.Timedelta(rule))
    return v.reindex(B.index + pd.Timedelta(f'{tf}min') - pd.Timedelta('1ms'), method='ffill').values


def build(tf, src='/home/claude/master/', thr=25.0):
    s = 2 / tf; r = np.sqrt(s); S = lambda k: max(1, int(round(k * s)))
    B = load(tf, src=src); o, h, l, c, v = (B[k].values.astype(float) for k in ('open', 'high', 'low', 'close', 'volume'))
    n = len(c); c1 = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1))); tr[0] = h[0] - l[0]
    atr = rma(tr, S(30))
    sh = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
    roll = lambda x, w, f: getattr(pd.Series(x).rolling(w), f)().values
    V = {}
    V['P04_htf4h'] = np.sign(np.nan_to_num(htf(B, tf, '4h', lambda H: H.close.values - z02.ema(H.close.values, 50))))
    dl = np.diff(c, prepend=c[0]); g = pd.Series(np.maximum(dl, 0)).ewm(alpha=1 / S(14), adjust=False).mean()
    ls = pd.Series(np.maximum(-dl, 0)).ewm(alpha=1 / S(14), adjust=False).mean(); rsi = (100 - 100 / (1 + g / ls)).values
    V['P07_rsi'] = np.where(rsi < 25, 1, np.where(rsi > 75, -1, 0))
    z = np.nan_to_num((c - z02.ema(c, S(240))) / atr); V['P08_zema'] = np.where(z <= -8 * r, 1, np.where(z >= 8 * r, -1, 0))
    m20 = np.nan_to_num((c - sh(c, S(20))) / atr)
    up_ = pd.Series((m20 >= 10 * r).astype(float)).rolling(S(60), min_periods=1).max().values
    dn_ = pd.Series((m20 <= -10 * r).astype(float)).rolling(S(60), min_periods=1).max().values
    V['P09_spike'] = dn_ - up_
    mu = roll(c, S(120), 'mean'); sd = roll(c, S(120), 'std'); V['P10_bb'] = np.where(c < mu - 2.5 * sd, 1, np.where(c > mu + 2.5 * sd, -1, 0))
    oi = B.oi_zec.values.astype(float); doi = (oi / sh(oi, S(30)) - 1) * 100; dp = (c / sh(c, S(30)) - 1) * 100
    V['P11_oiCont'] = np.where((doi > 1) & (dp > 0), 1, np.where((doi > 1) & (dp < 0), -1, 0))
    V['P12_oiLiq'] = np.where((doi < -2) & (dp < 0), 1, np.where((doi < -2) & (dp > 0), -1, 0))
    bt = B.btc_close.values.astype(float); dz = np.r_[0, np.diff(np.log(c))]; db = np.r_[0, np.diff(np.log(bt))]
    beta = np.nan_to_num(roll(dz * db, S(720), 'mean') / roll(db * db, S(720), 'mean')); res = roll(dz - beta * db, S(30), 'sum')
    rsd = roll(dz - beta * db, S(720), 'std') * np.sqrt(S(30)); q = np.nan_to_num(res / rsd)
    V['P15_resid'] = np.where(q > 2, 1, np.where(q < -2, -1, 0))
    bs = B.basis_bps.values.astype(float); bz = (bs - roll(bs, S(720), 'mean')) / roll(bs, S(720), 'std')
    V['X17_basis'] = np.where(bz <= -2, 1, np.where(bz >= 2, -1, 0))
    wb = roll(B.wh_buy_q50k.values - B.wh_sell_q50k.values, S(10), 'sum'); qv = roll(B.quote_volume.values, S(10), 'sum'); x = np.nan_to_num(wb / qv)
    V['R18_whale'] = np.where(x > 0.1, 1, np.where(x < -0.1, -1, 0))
    bi = np.nan_to_num(((B.book_m1 - B.book_p1) / (B.book_m1 + B.book_p1)).values); V['R19_book'] = np.where(bi > 0.2, 1, np.where(bi < -0.2, -1, 0))
    gl = B.glb_ls.values.astype(float); gz = (gl - roll(gl, S(5040), 'mean')) / roll(gl, S(5040), 'std')
    V['R21_glb'] = np.where(gz >= 2, -1, np.where(gz <= -2, 1, 0))
    tp = np.log(B.tt_pos_ls.values.astype(float)); d4 = np.nan_to_num(tp - sh(tp, S(120))); d1 = np.nan_to_num(tp - sh(tp, S(720)))
    V['ttposK4h'] = -np.sign(d4) * (np.abs(d4) > 0.03); V['ttposK1g'] = -np.sign(d1) * (np.abs(d1) > 0.06)
    qv30 = pd.Series(B.quote_volume.values).rolling(S(30)).sum().values
    wn = np.nan_to_num(pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(S(30)).sum().values / qv30)
    V['whaleT'] = np.sign(wn) * (np.abs(wn) > 0.05)
    adx = htf(B, tf, '4h', lambda H: regime.dmi(H)[0]); pdi = htf(B, tf, '4h', lambda H: regime.dmi(H)[1]); mdi = htf(B, tf, '4h', lambda H: regime.dmi(H)[2])
    RG = np.where(adx >= 20, np.sign(pdi - mdi), np.where(adx < 20, 0, 9))
    mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
    f240 = np.r_[c[S(240):] / c[:-S(240)] - 1, np.full(S(240), np.nan)] * 1e4
    order = ['ttposK4h', 'ttposK1g', 'whaleT', 'X17_basis', 'P09_spike', 'R21_glb', 'P07_rsi', 'P08_zema', 'P10_bb', 'P12_oiLiq', 'P11_oiCont', 'P15_resid', 'R18_whale', 'R19_book', 'P04_htf4h']
    L = np.zeros(n); Sh = np.zeros(n); rules = []
    for k in order:
        vv = np.asarray(V[k], float)
        for side in (1, -1):
            for lab, val in (('UP', 1), ('DN', -1), ('YAT', 0)):
                m = (vv == side) & (RG == val)
                if m[:mid].sum() < 200 * s: continue
                if np.nanmean((f240 * side)[:mid][m[:mid]]) >= thr:                    # IS seçimi
                    rules.append(f'{k}{"L" if side > 0 else "S"}@{lab}'); (L if side > 0 else Sh)[m] += 1
    return dict(B=B, o=o, h=h, l=l, c=c, atr=atr, s=s, r=r, S=S, L=L, Sh=Sh, rules=rules, mid=mid, RG=RG, n=n, V=V, f240=f240)


def run(D, kL=2, kS=3, ts=8.0, tH=480):
    z = np.zeros(D['n'], np.bool_); tL = D['L'] >= kL; tS = (D['Sh'] >= kS) & ~tL
    eq, T = z02.sim4(D['o'], D['h'], D['l'], D['c'], D['atr'], z, z, np.full(D['n'], np.nan), tL.astype(np.bool_), tS.astype(np.bool_),
                     float(ts * D['r']), int(D['S'](tH)), 99.0)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['et'] = D['B'].index[t.ei.astype(int)]
    return pd.Series(eq, index=D['B'].index), t


def ride(D, t, W=60, seed=5):
    """R2 girişlerinin canlı sürüş kalitesi, aynı yöndeki rastgele girişlere göre fark (W ve ATR süre eşdeğeri)."""
    Wb = D['S'](W); a = D['atr'] * D['r']; out = {}
    for half, sel in (('IS', t.ei < D['mid']), ('OOS', t.ei >= D['mid'])):
        res = []
        for side in (1, -1):
            e = t.ei[sel & (t.side == side)].astype(np.int64).values; e = e[e < D['n'] - Wb - 1]
            if len(e) < 20: continue
            P, rt, av, _ = path(D['h'], D['l'], D['c'], a, e, side, np.array([1.0, 2.0]), Wb)
            rng = np.random.default_rng(seed); lo, hi = (1000, D['mid']) if half == 'IS' else (D['mid'], D['n'] - Wb - 1)
            rnd = np.sort(rng.integers(lo, hi, 3000)).astype(np.int64)
            P0, rt0, av0, _ = path(D['h'], D['l'], D['c'], a, rnd, side, np.array([1.0, 2.0]), Wb)
            res.append((len(e), np.nanmean(P[:, 1]) * 100 - np.nanmean(P0[:, 1]) * 100, np.nanmean(rt) - np.nanmean(rt0), np.nanmean(av) - np.nanmean(av0)))
        w = np.array([x[0] for x in res], float); out[half] = tuple(np.average([x[k] for x in res], weights=w) for k in (1, 2, 3))
    return out
