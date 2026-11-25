# MİKRO KATMAN: V2.9 üstüne filtreli mikro gir-çık. Baz yönündeki aşırı uzamada (yukarı sıçrama long'da / aşağı sıçrama short'ta) kısa ters işlem,
# baz FLAT iken iki yönlü mikro ortalamaya dönüş (seçmeli). V2.9 uç TERS öncelikli. 500 $ sabit, piramit yok.
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, itertools, time
from numba import njit, prange
from pfsearch import o, h, l, c, posA, bXp, bXd, R10, tms, n, TICK, ix, S
Z = np.load('/home/claude/zecwork/base28.npz'); mid = int(Z['mid']); T0 = time.time()
vol = np.nan_to_num(Z['vol'].astype(float)); oi = S(Z['oi'].astype(float)).ffill().bfill().to_numpy(); btc = S(Z['btc'].astype(float)).ffill().bfill().to_numpy()
def rma(x, N): return S(x).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
pc = np.r_[c[0], c[:-1]]; AT = rma(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), 14)
vr = np.nan_to_num(vol / np.maximum(S(vol).rolling(1440, min_periods=60).mean().shift(1).to_numpy(), 1e-9))
def zsc(N): m = S(c).rolling(N).mean().to_numpy(); s = S(c).rolling(N).std().to_numpy(); return np.nan_to_num((c - m) / np.where(s > 0, s, np.nan))
def chg(x, k): return np.nan_to_num(x / np.r_[np.full(k, np.nan), x[:-k]] - 1)
oi15, b15 = chg(oi, 15), chg(btc, 15); rg = np.maximum(h - l, 1e-12); uw = (h - np.maximum(o, c)) / rg; lw = (np.minimum(o, c) - l) / rg
TRG = [(f'RSI10≥{X}', R10 >= X, R10 <= 100 - X) for X in (86, 88, 90, 92)]
for N in (60, 240):
    z = zsc(N); TRG += [(f'z{N}≥{zt:g}', z >= zt, z <= -zt) for zt in (2.5, 3.0, 3.5)]
TRG += [(f'iklim mum {k}ATR hacim≥{m}x', ((h - l) >= k * AT) & (c > o) & (vr >= m), ((h - l) >= k * AT) & (c < o) & (vr >= m)) for k in (3, 5) for m in (3, 5)]
FIL = [('filtresiz', np.ones(n, bool), np.ones(n, bool))] + [(f'hacim≥{m}x', vr >= m, vr >= m) for m in (2, 3, 5)]
FIL += [(f'OI15dk≤−%{x * 100:g}', oi15 <= -x, oi15 <= -x) for x in (0.005, 0.01)] + [(f'OI15dk≥+%{x * 100:g}', oi15 >= x, oi15 >= x) for x in (0.005, 0.01)]
FIL += [(f'BTC15dk |%|≤{x * 100:g}', np.abs(b15) <= x, np.abs(b15) <= x) for x in (0.003, 0.006)] + [(f'fitil≥{w:g}', uw >= w, lw >= w) for w in (0.3, 0.5)]
TU = np.stack([t[1] for t in TRG]); TD = np.stack([t[2] for t in TRG]); FU = np.stack([f[1] for f in FIL]); FD = np.stack([f[2] for f in FIL])
YV = np.array([50.0, 60.0, 70.0]); TBV = np.array([60, 240], np.int64); SV = np.array([0.01, 0.02])

@njit(cache=False)
def runM(eq, o, h, l, c, posA, bXp, bXd, R10, tu, td, fu, fd, Y, tb, s, flat, use, tick, out):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0
    mDir = 0; mPx = 0.0; mBar = 0; mStp = 0.0; isM = False; mp = 0.0; ml = 0.0; nm = 0; nt = 0; blk = 0; eq[0] = 0.0
    for i in range(1, n):
        if pos != 0:
            xp = -1.0; mstop = False
            if mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp): xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif mode == 3 and pos == mDir:
                if (pos == 1 and l[i] <= mStp) or (pos == -1 and h[i] >= mStp): xp = (min(o[i], mStp) - tick) if pos == 1 else (max(o[i], mStp) + tick); mode = 0; mstop = True
            elif mode == 0 and not np.isnan(bXp[i]) and pos == bXd[i]: xp = bXp[i]
            if xp > 0:
                pn = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pn; nt += 1
                if isM: mp += max(pn, 0.0); ml += max(-pn, 0.0)
                pos = 0; isM = False
                if mstop: blk = i + 30
        D = posA[i]; want = D
        if D != prevD: mode = 0
        else:
            if mode == 2:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
            elif mode == 3:
                ex = (R10[i] <= Y or i - mBar >= tb or c[i] >= mPx * 1.03) if mDir < 0 else (R10[i] >= 100.0 - Y or i - mBar >= tb or c[i] <= mPx * 0.97)
                if ex: mode = 0
                else: want = mDir if pos == mDir else 0
            else:
                if D != 0 and ((D > 0 and R10[i] >= 94.0) or (D < 0 and R10[i] <= 6.0)):
                    want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D
                elif use and i >= blk:
                    md = 0
                    if D > 0 and tu[i] and fu[i]: md = -1
                    elif D < 0 and td[i] and fd[i]: md = 1
                    elif D == 0 and flat:
                        if tu[i] and fu[i]: md = -1
                        elif td[i] and fd[i]: md = 1
                    if md != 0: want = md; mode = 3; mDir = md; mPx = c[i]; mBar = i; nm += 1
        prevD = D
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; pn = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pn; nt += 1
                if isM: mp += max(pn, 0.0); ml += max(-pn, 0.0)
                pos = 0; isM = False
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
                elif mode == 3 and want == mDir: mStp = ep * (1.0 - want * s); isM = True
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    out[13] = nm; out[14] = mp - ml; out[15] = mp / max(ml, 1e-9); out[16] = nt

@njit(cache=False)
def mets(eq, mid, t23, t300, B6, b6, BM, bm, out):
    n = len(eq); out[0] = eq[n - 1]; out[1] = eq[mid]; out[2] = eq[n - 1] - eq[mid]; out[3] = eq[n - 1] - eq[t23]; out[4] = eq[n - 1] - eq[t300]
    pk = -1e18; mdd = 0.0; m1 = 0.0; m2 = 0.0; s2 = 0.0; k = 0; pk2 = -1e18
    for i in range(0, n, 30):
        x = eq[i]; pk = max(pk, x); d = pk - x; mdd = max(mdd, d); s2 += (d / 5.0) ** 2; k += 1
        if i <= mid: m1 = max(m1, d)
        else: pk2 = max(pk2, x); m2 = max(m2, pk2 - x)
    out[5] = mdd; out[6] = m1; out[7] = m2; out[8] = np.sqrt(s2 / k); a = 0
    for j in range(len(B6) - 1):
        if eq[B6[j + 1]] - eq[B6[j]] - b6[j] > 0.5: a += 1
    out[9] = a; out[10] = len(B6) - 1; a = 0; z = 0
    for j in range(len(BM) - 1):
        r = eq[BM[j + 1]] - eq[BM[j]] - bm[j]
        if r > 0.5: a += 1
        elif r < -0.5: z += 1
    out[11] = a; out[12] = z

@njit(parallel=True, cache=False)
def grid(CB, o, h, l, c, posA, bXp, bXd, R10, TU, TD, FU, FD, YV, TBV, SV, tick, mid, t23, t300, B6, b6, BM, bm):
    R = np.zeros((len(CB), 17))
    for k in prange(len(CB)):
        eq = np.empty(len(c)); t, f = CB[k, 0], CB[k, 1]
        runM(eq, o, h, l, c, posA, bXp, bXd, R10, TU[t], TD[t], FU[f], FD[f], YV[CB[k, 2]], TBV[CB[k, 3]], SV[CB[k, 4]], CB[k, 5] == 1, CB[k, 6] == 1, tick, R[k])
        mets(eq, mid, t23, t300, B6, b6, BM, bm, R[k])
    return R

def bidx(freq):
    dr = pd.date_range(ix[0].normalize(), ix[-1], freq=freq); b = np.searchsorted(ix.values, dr.values); return np.unique(np.r_[0, b[(b > 0) & (b < n)], n - 1]).astype(np.int64)
B6, BM = bidx('6MS'), bidx('MS')
t23 = int(np.searchsorted(tms, (pd.Timestamp('2023-02-06') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'))); t300 = int(np.searchsorted(tms, tms[-1] - 300 * 86400000))
eq0 = np.load('/home/claude/zecwork/eq_V29_yatay_baz.npy'); b6, bm = np.diff(eq0[B6]), np.diff(eq0[BM])
CB = np.array([(0, 0, 0, 0, 0, 0, 0)] + [(t, f, y, tb, s_, fl, 1) for t in range(len(TRG)) for f in range(len(FIL)) for y, tb, s_, fl in itertools.product(range(3), range(2), range(2), range(2))], np.int64)
if __name__ == '__main__':
    T1 = time.time(); R = grid(CB, o, h, l, c, posA, bXp, bXd, R10, TU, TD, FU, FD, YV, TBV, SV, TICK, mid, t23, t300, B6, b6, BM, bm)
    cols = ['net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'mddIS', 'mddOOS', 'Ulcer', 'blok+', 'blok', 'ay+', 'ay-', 'mikro', 'mikroNet', 'mikroPF', 'işlem']
    df = pd.DataFrame(R, columns=cols); df['tetik'] = [TRG[k][0] for k in CB[:, 0]]; df['filtre'] = [FIL[k][0] for k in CB[:, 1]]
    df['çıkışRSI'] = YV[CB[:, 2]]; df['süre'] = TBV[CB[:, 3]]; df['stop'] = SV[CB[:, 4]]; df['flatta'] = CB[:, 5]
    b = df.iloc[0]
    for k in ('net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'Ulcer'): df['d' + k] = df[k] - b[k]
    df.to_pickle('/home/claude/zecwork/mikro.pkl'); print(f'baz {b.net:.1f} (9061.9 olmalı) işlem {b.işlem:.0f} | {len(CB) - 1} varyant {time.time() - T1:.0f} sn')
