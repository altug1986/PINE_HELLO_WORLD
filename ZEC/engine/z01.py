"""CYBORG | ZEC 2M ÇİFT YÖN V0.1 — Pine'ın birebir araştırma portu (TV değil).

Kurallar (Pine ZEC/pine/ZEC_2M_V0.1.pine ile aynı):
  ATR = RMA(TR, 30); hacim oranı = volume / SMA(volume, 360)[1]
  E1 İTKİ   : close > highest(high,N1)[1] & (close-close[1]) >= k1*ATR & hacim oranı >= v1   (short ayna)
  E2 YAPI   : close > highest(high,N2)[1]                                                    (short ayna)
  Giriş/ters çevirme bar kapanışında (+/- 1 tick kayma). Çıkış: stop emri (ilk SL s*ATR, iz = en iyi kapanış -/+ m*ATR,
  sadece sıkılaşır), bir sonraki barda tetiklenir, boşlukta açılıştan. Komisyon %0,04 her bacak, sabit 500 $.
"""
import numpy as np, pandas as pd
from numba import njit

FEE, TICK, CAP = 0.0004, 0.01, 500.0


def rma(x, n):
    a = np.full(len(x), np.nan); s = np.nansum(x[:n]) / n; a[n - 1] = s
    al = 1.0 / n
    for i in range(n, len(x)):
        s = al * x[i] + (1 - al) * s; a[i] = s
    return a


def prep(B):
    o, h, l, c, v = (B[k].values.astype(np.float64) for k in ('open', 'high', 'low', 'close', 'volume'))
    c1 = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1))); tr[0] = h[0] - l[0]
    F = dict(o=o, h=h, l=l, c=c, atr=rma(tr, 30),
             vma=pd.Series(v).rolling(360).mean().shift(1).values, v=v)
    F['HH'] = {}; F['LL'] = {}
    return F


def hl(F, n):
    if n not in F['HH']:
        F['HH'][n] = pd.Series(F['h']).rolling(n).max().shift(1).values
        F['LL'][n] = pd.Series(F['l']).rolling(n).min().shift(1).values
    return F['HH'][n], F['LL'][n]


@njit(cache=True)
def sim(o, h, l, c, atr, v, vma, HH1, LL1, HH2, LL2, k1, v1, m, s, useE1, useE2):
    n = len(c); eq = np.zeros(n); real = 0.0
    pos = 0; ep = 0.0; q = 0.0; best = 0.0; stop = 0.0; ei = 0; own = 0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    mae = 0.0
    for i in range(1, n):
        # 1) stop emri (önceki barda kuruldu)
        if pos == 1 and l[i] <= stop:
            xp = min(o[i], stop) - TICK
            pnl = q * (xp - ep) - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = 1; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = max(mae, (ep - l[i]) / ep); T[nt, 7] = own; nt += 1
            pos = 0
        elif pos == -1 and h[i] >= stop:
            xp = max(o[i], stop) + TICK
            pnl = q * (ep - xp) - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = -1; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = max(mae, (h[i] - ep) / ep); T[nt, 7] = own; nt += 1
            pos = 0
        elif pos == 1:
            mae = max(mae, (ep - l[i]) / ep)
        elif pos == -1:
            mae = max(mae, (h[i] - ep) / ep)
        # 2) sinyaller (bar kapanışı)
        a = atr[i]
        if np.isnan(a) or np.isnan(vma[i]) or np.isnan(HH2[i]) or np.isnan(HH1[i]):
            eq[i] = real + (q * (c[i] - ep) * pos if pos != 0 else 0.0); continue
        d = c[i] - c[i - 1]
        e1L = useE1 and c[i] > HH1[i] and d >= k1 * a and v[i] >= v1 * vma[i]
        e1S = useE1 and c[i] < LL1[i] and -d >= k1 * a and v[i] >= v1 * vma[i]
        e2L = useE2 and c[i] > HH2[i]
        e2S = useE2 and c[i] < LL2[i]
        wL = e1L or e2L; wS = e1S or e2S
        if wL and wS:
            wL = False; wS = False
        want = 1 if wL else (-1 if wS else 0)
        if want != 0 and want != pos:
            if pos != 0:   # ters çevirme: önce kapat
                xp = c[i] - TICK * pos
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = want; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; best = c[i]; mae = 0.0
            own = 1 if (e1L if pos == 1 else e1S) else 2
            stop = ep - pos * s * a
        # 3) iz stop (sadece sıkılaşır)
        if pos == 1:
            best = max(best, c[i]); stop = max(stop, best - m * a)
        elif pos == -1:
            best = min(best, c[i]); stop = min(stop, best + m * a)
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


def run(B, F, N1=60, k1=2.5, v1=3.0, N2=180, m=6.0, s=6.0, E1=True, E2=True):
    HH1, LL1 = hl(F, N1); HH2, LL2 = hl(F, N2)
    eq, T = sim(F['o'], F['h'], F['l'], F['c'], F['atr'], F['v'], F['vma'], HH1, LL1, HH2, LL2,
                k1, v1, m, s, E1, E2)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
    t['et'] = B.index[t.ei.astype(int)]; t['xt'] = B.index[t.xi.astype(int)]
    return pd.Series(eq, index=B.index), t


def metrics(eq, t, lo=None, hi=None):
    e = eq.loc[lo:hi]; e = e - e.iloc[0]
    tt = t[(t.et >= e.index[0]) & (t.et <= e.index[-1])]
    m30 = e.resample('30min').last().dropna(); dd = m30.cummax() - m30
    days = (e.index[-1] - e.index[0]).total_seconds() / 86400; yr = e.iloc[-1] * 365 / days
    dly = e.resample('1D').last().dropna().diff().dropna()
    g = tt.pnl.clip(lower=0).sum(); ls = -tt.pnl.clip(upper=0).sum()
    s10 = e.resample('10min').last().dropna().diff().std() / CAP * 100
    s1h = e.resample('1h').last().dropna().diff().std() / CAP * 100
    mon = e.resample('ME').last().diff().fillna(e.resample('ME').last().iloc[:1])
    return dict(net=e.iloc[-1], pct=e.iloc[-1] / CAP * 100, n=len(tt), PF=g / ls if ls else np.nan,
                win=(tt.pnl > 0).mean() * 100, bps=(tt.pnl / CAP * 1e4).mean(),
                fee=(FEE * CAP / tt.ep * (tt.ep + tt.xp)).sum(), MaxDD=dd.max(),
                P95=np.percentile(dd, 95), P98=np.percentile(dd, 98), P99=np.percentile(dd, 99),
                Calmar=yr / dd.max() if dd.max() > 0 else np.nan,
                Sharpe=dly.mean() / dly.std() * np.sqrt(365), Sortino=dly.mean() / dly[dly < 0].std() * np.sqrt(365),
                Ulcer=np.sqrt((dd ** 2).mean()), s10=s10, s1h=s1h, posM=(mon > 0).mean() * 100,
                maeP90=np.percentile(tt.mae, 90) * 100 if len(tt) else np.nan, maeMax=tt.mae.max() * 100 if len(tt) else np.nan)


def sc4(eq):
    """SUPER-CALMAR V4 katman 1 (10 dk MTM, sabit nominal %)."""
    e = eq.resample('10min').last().dropna() / CAP * 100; e = e - e.iloc[0]
    days = (e.index[-1] - e.index[0]).total_seconds() / 86400; yr = e.iloc[-1] * 365 / days
    dd = e.cummax() - e; mdd = dd.max()
    d = e.resample('1D').last().dropna()
    r30 = [(d.iloc[max(0, j - 30):j + 1].cummax() - d.iloc[max(0, j - 30):j + 1]).max() for j in range(len(d))]
    r30 = np.array(r30); es = r30[r30 >= np.percentile(r30, 95)].mean()
    dtail = (np.percentile(r30, 95) + np.percentile(r30, 99) + es) / 3
    smul = (np.sqrt(144) * e.diff().std() + np.sqrt(24) * e.resample('1h').last().diff().std() + d.diff().std()) / 3
    sc3 = yr / (mdd + dtail + smul)
    sec = (e.index - pd.Timestamp(0)).total_seconds().values          # birimden bağımsız (ns/us)
    ath = e.cummax(); last = pd.Series(np.where(e >= ath, sec, np.nan), index=e.index).ffill()
    a95 = np.percentile((sec - last.values) / 86400, 95)
    r7 = d.diff(7).dropna(); mu = r7.mean()
    qd = 0.0 if mu <= 0 else 1 / (2 - min(1, np.percentile(r7, 10) / mu))
    w = d.resample('W').last().diff().dropna(); pw = w[w > 0].sort_values(ascending=False)
    K = pw.iloc[:max(1, int(np.ceil(len(pw) * 0.1)))].sum() / pw.sum() if len(pw) else 1
    return dict(SC3=sc3, SC4=sc3 * np.sqrt(1 / (1 + a95 / 7)) * qd * (1 - K / 2), A95=a95, Qd=qd, K=K)


def events(B, thr=7.0, bars=10):
    c = B.close.values; r = (c[bars:] / c[:-bars] - 1) * 100
    idx = np.where(np.abs(r) >= thr)[0] + bars; out = []; last = -10**9
    for i in idx:
        if i - last > 720:
            out.append((B.index[i - bars], np.sign(r[i - bars])))
            last = i
    return out


# ---------------- V0.1 çekirdek: SPIKE FADE (olay çalışmasından: 10 bar ≥10 ATR fade, IS/OOS +62 bps h60) ----------------
@njit(cache=True)
def sim2(o, h, l, c, atr, sigL, sigS, s, m, H, own_id):
    n = len(c); eq = np.zeros(n); real = 0.0
    pos = 0; ep = 0.0; q = 0.0; best = 0.0; stop = 0.0; ei = 0; ae = 0.0; mae = 0.0; own = 0
    T = np.zeros((n // 2 + 10, 8)); nt = 0
    for i in range(1, n):
        if pos != 0:
            hit = (pos == 1 and l[i] <= stop) or (pos == -1 and h[i] >= stop)
            if hit:
                xp = (min(o[i], stop) - TICK) if pos == 1 else (max(o[i], stop) + TICK)
                pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
                pos = 0
            else:
                mae = max(mae, (ep - l[i]) / ep if pos == 1 else (h[i] - ep) / ep)
        want = 1 if sigL[i] else (-1 if sigS[i] else 0)
        tout = pos != 0 and i - ei >= H
        blk = 0
        if pos != 0 and ((want != 0 and want != pos) or tout):
            blk = pos if (tout and want == pos) else 0      # Pine paritesi: zaman çıkışı barında aynı yöne yeniden giriş yok
            xp = c[i] - TICK * pos
            pnl = q * (xp - ep) * pos - FEE * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = mae; T[nt, 7] = own; nt += 1
            pos = 0
        if pos == 0 and want != 0 and want != blk and not np.isnan(atr[i]):
            pos = want; ep = c[i] + TICK * pos; q = CAP / ep; ei = i; best = c[i]; ae = atr[i]; mae = 0.0; own = own_id[i]
            stop = ep - pos * s * ae
        if pos == 1:
            best = max(best, c[i]); stop = max(stop, best - m * ae)
        elif pos == -1:
            best = min(best, c[i]); stop = min(stop, best + m * ae)
        eq[i] = real + (q * (c[i] - ep) * pos - FEE * q * ep if pos != 0 else 0.0)
    return eq, T[:nt]


def fade_sig(F, bars, k):
    c, a = F['c'], F['atr']
    z = (c - np.r_[np.full(bars, np.nan), c[:-bars]]) / a
    return np.nan_to_num(z <= -k).astype(np.bool_), np.nan_to_num(z >= k).astype(np.bool_)


def run2(B, F, bars=10, k=10, s=6, m=99, H=120):
    sL, sS = fade_sig(F, bars, k)
    eq, T = sim2(F['o'], F['h'], F['l'], F['c'], F['atr'], sL, sS, s, m, H, np.ones(len(sL)))
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own'])
    t['et'] = B.index[t.ei.astype(int)]; t['xt'] = B.index[t.xi.astype(int)]
    return pd.Series(eq, index=B.index), t
