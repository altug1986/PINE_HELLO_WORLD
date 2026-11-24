"""ZEC 2M OY KONSORSİYUMU — ham oylama (filtresiz) → sonra seçim.

Bileşen = her bar kapanışında +1 / 0 / −1 oy. Sınıf: P = Pine'da birebir, X = Pine'da vekil, R = yalnız araştırma.
Skor = Σ w_j · oy_j. Giriş skor ≥ Tin (long) / ≤ −Tin (short); çıkış skor < Tout (long) / > −Tout (short);
karşı giriş eşiği aşılırsa ters çevir; felaket stopu s·ATR (giriş ATR'si). Komisyon %0,04, kayma 1 tick, sabit 500 $.
"""
import numpy as np, pandas as pd
from numba import njit
import z01

FEE, TICK, CAP = z01.FEE, z01.TICK, z01.CAP


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def htf(B, rule, fn):
    """Kapanmış HTF barından türetilen seri → 2 dk bar kapanışına (HTF kapanışı <= T)."""
    H = B[['open', 'high', 'low', 'close']].resample(rule, label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    v = pd.Series(fn(H), index=H.index + pd.Timedelta(rule))          # HTF kapanış anı
    T = B.index + pd.Timedelta('2min') - pd.Timedelta('1ms')          # Pine paritesi: HTF[1] + lookahead_on (sınır barında önceki HTF)
    return v.reindex(T, method='ffill').values


def sgn(x, band=0.0):
    return np.where(x > band, 1, np.where(x < -band, -1, 0)).astype(np.int8)


def votes(B, X1=None):
    c, h, l = B.close.values, B.high.values, B.low.values
    F = z01.prep(B); a = F['atr']; V = {}
    V['P01_ema'] = sgn(ema(c, 20) - ema(c, 60))
    # Supertrend (ATR RMA30, çarpan 3)
    hl2 = (h + l) / 2; up = hl2 - 3 * a; dn = hl2 + 3 * a; d = np.ones(len(c), np.int8)
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        d[i] = 1 if c[i] > fd[i - 1] else (-1 if c[i] < fu[i - 1] else d[i - 1])
    V['P02_st'] = d
    V['P03_htf30'] = sgn(np.nan_to_num(htf(B, '30min', lambda H: np.r_[np.nan, np.diff(ema(H.close.values, 20))])))
    V['P04_htf4h'] = sgn(np.nan_to_num(htf(B, '4h', lambda H: H.close.values - ema(H.close.values, 50))))
    HH, LL = z01.hl(F, 120); V['P05_don'] = sgn(np.nan_to_num(c - (HH + LL) / 2))
    r60 = c - np.r_[np.full(60, np.nan), c[:-60]]; V['P06_roc'] = sgn(np.nan_to_num(r60 / a), 1.0)
    dlt = np.diff(c, prepend=c[0]); g = pd.Series(np.maximum(dlt, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    ls = pd.Series(np.maximum(-dlt, 0)).ewm(alpha=1 / 14, adjust=False).mean(); rsi = (100 - 100 / (1 + g / ls)).values
    V['P07_rsi'] = np.where(rsi < 25, 1, np.where(rsi > 75, -1, 0)).astype(np.int8)
    z = np.nan_to_num((c - ema(c, 240)) / a); V['P08_zema'] = np.where(z <= -8, 1, np.where(z >= 8, -1, 0)).astype(np.int8)
    m20 = np.nan_to_num((c - np.r_[np.full(20, np.nan), c[:-20]]) / a)
    up_ = pd.Series((m20 >= 10).astype(float)).rolling(60, min_periods=1).max().values
    dn_ = pd.Series((m20 <= -10).astype(float)).rolling(60, min_periods=1).max().values
    V['P09_spike'] = (dn_ - up_).astype(np.int8)
    mu = pd.Series(c).rolling(120).mean().values; sd = pd.Series(c).rolling(120).std().values
    V['P10_bb'] = np.where(c < mu - 2.5 * sd, 1, np.where(c > mu + 2.5 * sd, -1, 0)).astype(np.int8)
    oi = B.oi_zec.values; doi = (oi / np.r_[np.full(30, np.nan), oi[:-30]] - 1) * 100
    dp = (c / np.r_[np.full(30, np.nan), c[:-30]] - 1) * 100
    V['P11_oiCont'] = np.where((doi > 1) & (dp > 0), 1, np.where((doi > 1) & (dp < 0), -1, 0)).astype(np.int8)
    V['P12_oiLiq'] = np.where((doi < -2) & (dp < 0), 1, np.where((doi < -2) & (dp > 0), -1, 0)).astype(np.int8)
    bt = B.btc_close.values; V['P13_btc'] = sgn(np.nan_to_num((bt / np.r_[np.full(30, np.nan), bt[:-30]] - 1) * 100), 0.1)
    xm = B.xmr_close.values; V['P14_xmr'] = sgn(np.nan_to_num((xm / np.r_[np.full(30, np.nan), xm[:-30]] - 1) * 100), 0.2)
    rz = np.log(c); rb = np.log(bt); dz = np.r_[0, np.diff(rz)]; db = np.r_[0, np.diff(rb)]
    cov = pd.Series(dz * db).rolling(720).mean().values; var = pd.Series(db * db).rolling(720).mean().values
    beta = np.nan_to_num(cov / var); res = pd.Series(dz - beta * db).rolling(30).sum().values
    rsd = pd.Series(dz - beta * db).rolling(720).std().values * np.sqrt(30)
    V['P15_resid'] = sgn(np.nan_to_num(res / rsd), 2.0)
    # X: Pine vekili — 1 dk mum yönüne göre hacim farkı (request.security_lower_tf ile aynı)
    if X1 is not None:
        sv = (np.sign(X1.close - X1.open) * X1.volume).resample('2min', label='left', closed='left').sum().reindex(B.index).values
        fl = pd.Series(sv).rolling(10).sum().values / pd.Series(B.volume.values).rolling(10).sum().values
        V['X16_flow'] = sgn(np.nan_to_num(fl), 0.15)
    bs = B.basis_bps.values; bz = (bs - pd.Series(bs).rolling(720).mean().values) / pd.Series(bs).rolling(720).std().values
    V['X17_basis'] = np.where(bz <= -2, 1, np.where(bz >= 2, -1, 0)).astype(np.int8)
    # R: yalnız araştırma (TV'de yok)
    wb = pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(10).sum().values
    qv = pd.Series(B.quote_volume.values).rolling(10).sum().values
    V['R18_whale'] = sgn(np.nan_to_num(wb / qv), 0.1)
    bi = ((B.book_m1 - B.book_p1) / (B.book_m1 + B.book_p1)).values; V['R19_book'] = sgn(np.nan_to_num(bi), 0.2)
    tp = np.log(B.tt_pos_ls.values); V['R20_ttpos'] = sgn(np.nan_to_num((tp - np.r_[np.full(30, np.nan), tp[:-30]]) * 100), 3)
    gl = B.glb_ls.values; gz = (gl - pd.Series(gl).rolling(5040).mean().values) / pd.Series(gl).rolling(5040).std().values
    V['R21_glb'] = np.where(gz >= 2, -1, np.where(gz <= -2, 1, 0)).astype(np.int8)
    names = list(V); M = np.stack([V[k] for k in names], 1).astype(np.int8)
    return names, M, F


@njit(cache=True)
def sim3(o, h, l, c, atr, S, Tin, Tout, s, H):
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; mae = 0.0
    T = np.zeros((n // 2 + 10, 7)); nt = 0
    for i in range(1, n):
        if pos != 0:
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; nt += 1
                pos = 0
            else:
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
        sc = S[i]
        want = 1 if sc >= Tin else (-1 if sc <= -Tin else 0)
        ex = pos != 0 and ((pos == 1 and sc < Tout) or (pos == -1 and sc > -Tout) or (want != 0 and want != pos) or (i - ei >= H))
        blk = pos if (ex and i - ei >= H and want == pos) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; nt += 1
            pos = 0
        if pos == 0 and want != 0 and want != blk and not np.isnan(atr[i]):
            pos = want; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; mae = 0.0; stop = ep - pos * s * atr[i]
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


def run3(B, F, M, w, Tin, Tout, s=6.0, H=10**9):
    S = (M.astype(np.float64) @ np.asarray(w, np.float64))
    eq, T = sim3(F['o'], F['h'], F['l'], F['c'], F['atr'], S, float(Tin), float(Tout), float(s), int(H))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae'])
    t['et'] = B.index[t.ei.astype(int)]; t['xt'] = B.index[t.xi.astype(int)]
    return pd.Series(eq, index=B.index), t


# ---------------- ÇEKİRDEK TREND + TAKTİK (sahip durum makinesi) ----------------
@njit(cache=True)
def sim4(o, h, l, c, atr, eL, eS, D, tL, tS, ts, tH, tD):
    """Sahip 1 = TREND (giriş eL/eS, iz D[i] fiyat birimi, süre sınırı yok).
       Sahip 2 = TAKTİK (yalnız trend boşken; giriş tL/tS, ilk SL ts*ATR, iz tD*ATR, süre tH bar).
       Trend sinyali aynı yöndeki taktik pozisyonu devralır (giriş fiyatı korunur), ters yönde çevirir."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; own = 0; ep = 0.0; q = 0.0; stop = 0.0; best = 0.0; ei = 0; mae = 0.0; ae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    for i in range(1, n):
        if pos != 0:
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = 0
            else:
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
        wT = 1 if eL[i] else (-1 if eS[i] else 0)
        wK = 1 if tL[i] else (-1 if tS[i] else 0)
        # çıkış/çevirme kararları (bar kapanışı)
        ex = False
        if pos != 0 and wT != 0 and wT != pos:
            ex = True
        elif pos != 0 and own == 2 and ((wK != 0 and wK != pos) or i - ei >= tH):
            ex = True
        blk = pos if (ex and own == 2 and i - ei >= tH and (wK == pos or wT == pos)) else 0   # Pine paritesi
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = 0
        if pos != 0 and own == 2 and wT == pos:      # trend devralır
            own = 1; stop = (max(stop, best - D[i]) if pos == 1 else min(stop, best + D[i]))
        if pos == 0 and not np.isnan(atr[i]):
            if wT != 0 and wT != blk and not np.isnan(D[i]):
                pos = wT; own = 1; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; mae = 0.0; best = c[i]; stop = ep - pos * D[i]
            elif wK != 0 and wK != blk:
                pos = wK; own = 2; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; mae = 0.0; best = c[i]; ae = atr[i]; stop = ep - pos * ts * ae
        if pos != 0:
            dist = D[i] if own == 1 else tD * ae
            if pos == 1:
                best = max(best, c[i]); stop = max(stop, best - dist)
            else:
                best = min(best, c[i]); stop = min(stop, best + dist)
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


@njit(cache=True)
def sim5(o, h, l, c, atr, aL, aS, ats, atH, bL, bS, bts, btH):
    """İki motor, öncelik A > B. A sinyali: aynı yönde B pozisyonunu devralır (giriş fiyatı korunur, A zarar kesi ve süresi
    devralma anından), ters yönde çevirir. B yalnız boşken ya da kendi pozisyonunu çevirirken işlem açar. Her motorun
    kendi zarar kesi (ATR katı, giriş ATR'si) ve süre sınırı (bar) vardır."""
    n = len(c); eq = np.zeros(n); real = 0.0; pos = 0; own = 0; ep = 0.0; q = 0.0; stop = 0.0; ei = 0; ti = 0; mae = 0.0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    for i in range(1, n):
        if pos != 0:
            if (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop):
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0; own = 0
            else:
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
        wA = 1 if aL[i] else (-1 if aS[i] else 0); wB = 1 if bL[i] else (-1 if bS[i] else 0)
        tout = pos != 0 and ((own == 1 and i - ti >= atH) or (own == 2 and i - ti >= btH))
        ex = pos != 0 and ((wA != 0 and wA != pos) or (own == 2 and wB != 0 and wB != pos) or tout)
        blk = pos if (tout and (wA == pos or wB == pos)) else 0
        if ex:
            xp = c[i] - TICK * pos; pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0; own = 0
        if pos != 0 and own == 2 and wA == pos:            # A devralır
            own = 1; ti = i; s2 = c[i] - pos * ats * atr[i]
            stop = max(stop, s2) if pos == 1 else min(stop, s2)
        if pos == 0 and not np.isnan(atr[i]):
            if wA != 0 and wA != blk:
                pos = wA; own = 1; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * ats * atr[i]
            elif wB != 0 and wB != blk:
                pos = wB; own = 2; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; ti = i; mae = 0.0; stop = ep - pos * bts * atr[i]
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]
