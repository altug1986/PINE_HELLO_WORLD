# YATAY MOD: trend sonrası yatay tespit edilince pozisyonu "kaymak" noktada kapat (short: dipten dönüş, long: tepeden dönüş),
# geri çekilmede / devamda / süre dolunca / yatay bitince baz yöne yeniden gir. V2.9 uç TERS kuralı öncelikli ve aynen duruyor.
import sys; sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P; import numpy as np, pandas as pd, itertools, time
from numba import njit, prange
from pfsearch import o, h, l, c, posA, bXp, bXd, R10, tms, n, TICK, ix, S
mid = int(np.load('/home/claude/zecwork/base28.npz')['mid']); T0 = time.time()
def rma(x, N): return S(x).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
def htf(rule): return pd.DataFrame({'h': h, 'l': l, 'c': c}, ix).resample(rule).agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
def trng(D):
    H, L, C = D.h.values, D.l.values, D.c.values; pc = np.r_[C[0], C[:-1]]
    return H, L, np.maximum(H - L, np.maximum(np.abs(H - pc), np.abs(L - pc)))
def adx(D, N=14):
    H, L, tr = trng(D); up = np.r_[0, np.diff(H)]; dn = np.r_[0, -np.diff(L)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0); at = rma(tr, N)
    pdi = 100 * rma(pdm, N) / at; mdi = 100 * rma(mdm, N) / at
    return rma(100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, 1, pdi + mdi), N)
def chop(D, N=14):
    H, L, tr = trng(D); s = S(tr).rolling(N).sum().values; rg = S(H).rolling(N).max().values - S(L).rolling(N).min().values
    return 100 * np.log10(s / np.maximum(rg, 1e-12)) / np.log10(N)
def mapH(v, D, td):   # HTF bar KAPANINCA kullanılır (Pine: request.security(..., x[1], lookahead_on))
    s_ = S(v, D.index + td); j = np.searchsorted(s_.index.values, ix.values, side='right') - 1; out = s_.values[np.maximum(j, 0)]; out[j < 0] = np.nan; return out
D15, D1, D4 = htf('15min'), htf('1h'), htf('4h')
A1, A4 = mapH(adx(D1), D1, pd.Timedelta('1h')), mapH(adx(D4), D4, pd.Timedelta('4h'))
C15, C1 = mapH(chop(D15), D15, pd.Timedelta('15min')), mapH(chop(D1), D1, pd.Timedelta('1h'))
DET = [('YOK', np.zeros(n, bool)), ('ADX 1s<20', A1 < 20), ('ADX 1s<25', A1 < 25), ('ADX 4s<20', A4 < 20), ('ADX 4s<25', A4 < 25),
       ('CHOP 15dk>55', C15 > 55), ('CHOP 15dk>61.8', C15 > 61.8), ('CHOP 1s>55', C1 > 55), ('CHOP 1s>61.8', C1 > 61.8), ('HER ZAMAN', np.ones(n, bool))]
pc = np.r_[c[0], c[:-1]]; AT = rma(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), 14)
def rmin(x, N): return S(x).rolling(N, min_periods=1).min().to_numpy()
def rmax(x, N): return S(x).rolling(N, min_periods=1).max().to_numpy()
lowk, highk = rmin(l, 15), rmax(h, 15); EX = []
for Nr in (60, 240):
    LL, HH = rmin(l, Nr), rmax(h, Nr); W = HH - LL
    for Pex in (0.8, 0.9):
        for m in (0.5, 1.0, 2.0):
            EX.append((f'kanal{Nr} dip/tepe %{(1 - Pex) * 100:g} + {m:g}ATR dönüş', (lowk <= LL + (1 - Pex) * W) & (c >= lowk + m * AT), (highk >= HH - (1 - Pex) * W) & (c <= highk - m * AT)))
for L in (14, 30, 60):
    rs = np.nan_to_num(P.rsi(c, L), nan=50.0); rp = np.r_[50.0, rs[:-1]]
    for th in (25.0, 30.0):
        EX.append((f'RSI{L} {th:g} geri dönüş', (rp < th) & (rs >= th), (rp > 100 - th) & (rs <= 100 - th)))
LL, HH = rmin(l, 240), rmax(h, 240); PB = np.where(HH - LL > 0, (c - LL) / np.maximum(HH - LL, 1e-12), 0.5)
YAT = np.stack([d[1] for d in DET]); EXS = np.stack([e[1] for e in EX]); EXL = np.stack([e[2] for e in EX])
PRE = np.array([0.5, 0.75]); CONT = np.array([0.01, 0.02, 0.04]); TBV = np.array([240, 960], np.int64); GV = np.array([-9.0, 0.0, 0.02])

@njit(cache=False)
def run1(eq, o, h, l, c, posA, bXp, bXd, R10, yat, exS, exL, PB, Pre, cont, tb, flip, g, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0
    rm = 0; xPx = 0.0; xBar = 0; xSide = 0; fStp = 0.0; nx = 0; eq[0] = 0.0
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if mode == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp): xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode = 0
            elif rm == 1 and pos == -xSide:
                if (pos == 1 and l[i] <= fStp) or (pos == -1 and h[i] >= fStp): xp = (min(o[i], fStp) - tick) if pos == 1 else (max(o[i], fStp) + tick)
            elif mode == 0 and not np.isnan(bXp[i]) and pos == bXd[i]: xp = bXp[i]
            if xp > 0: real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD: mode = 0; rm = 0
        elif D != 0:
            if mode == 0:
                if (D > 0 and R10[i] >= 94.0) or (D < 0 and R10[i] <= 6.0):                       # V2.9 uç TERS (öncelikli)
                    want = -D; mode = 2; tPx = c[i]; tBar = i; tSide = D; rm = 0
                elif rm == 0:
                    if yat[i] and pos == D and ((D < 0 and exS[i]) or (D > 0 and exL[i])) and (g < -1.0 or (c[i] / ep - 1.0) * D >= g):
                        rm = 1; xPx = c[i]; xBar = i; xSide = D; nx += 1; want = -D if flip else 0
                else:
                    re = (i - xBar >= tb) or (not yat[i]) or ((PB[i] >= Pre) if D < 0 else (PB[i] <= 1.0 - Pre)) or ((c[i] <= xPx * (1.0 - cont)) if D < 0 else (c[i] >= xPx * (1.0 + cont)))
                    if re: rm = 0
                    else: want = -D if (flip and pos == -D) else 0
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        if want != pos:
            if pos != 0: xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if mode == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
                elif rm == 1 and want == -xSide: fStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return nx

@njit(cache=False)
def mets(eq, mid, t23, t300, B6, b6, BM, bm, out):
    n = len(eq); out[0] = eq[n - 1]; out[1] = eq[mid]; out[2] = eq[n - 1] - eq[mid]; out[3] = eq[n - 1] - eq[t23]; out[4] = eq[n - 1] - eq[t300]
    pk = -1e18; mdd = 0.0; m1 = 0.0; m2 = 0.0; s2 = 0.0; k = 0; pk2 = -1e18
    for i in range(0, n, 30):
        x = eq[i]; pk = max(pk, x); d = pk - x; mdd = max(mdd, d); s2 += (d / 5.0) ** 2; k += 1
        if i <= mid: m1 = max(m1, d)
        else: pk2 = max(pk2, x); m2 = max(m2, pk2 - x)
    out[5] = mdd; out[6] = m1; out[7] = m2; out[8] = np.sqrt(s2 / k)
    a = 0
    for j in range(len(B6) - 1):
        if eq[B6[j + 1]] - eq[B6[j]] - b6[j] > 0.5: a += 1
    out[9] = a; out[10] = len(B6) - 1; a = 0; z = 0
    for j in range(len(BM) - 1):
        r = eq[BM[j + 1]] - eq[BM[j]] - bm[j]
        if r > 0.5: a += 1
        elif r < -0.5: z += 1
    out[11] = a; out[12] = z

@njit(parallel=True, cache=False)
def grid(CB, PRE, CONT, TBV, GV, o, h, l, c, posA, bXp, bXd, R10, YAT, EXS, EXL, PB, tick, mid, t23, t300, B6, b6, BM, bm):
    R = np.zeros((len(CB), 14))
    for k in prange(len(CB)):
        eq = np.empty(len(c))
        nx = run1(eq, o, h, l, c, posA, bXp, bXd, R10, YAT[CB[k, 0]], EXS[CB[k, 1]], EXL[CB[k, 1]], PB, PRE[CB[k, 2]], CONT[CB[k, 3]], TBV[CB[k, 4]], CB[k, 5] == 1, GV[CB[k, 6]], tick)
        mets(eq, mid, t23, t300, B6, b6, BM, bm, R[k]); R[k, 13] = nx
    return R

def bidx(freq):
    dr = pd.date_range(ix[0].normalize(), ix[-1], freq=freq); b = np.searchsorted(ix.values, dr.values); return np.unique(np.r_[0, b[(b > 0) & (b < n)], n - 1]).astype(np.int64)
B6, BM = bidx('6MS'), bidx('MS')
t23 = int(np.searchsorted(tms, (pd.Timestamp('2023-02-06') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'))); t300 = int(np.searchsorted(tms, tms[-1] - 300 * 86400000))
eq0 = np.empty(n); run1(eq0, o, h, l, c, posA, bXp, bXd, R10, YAT[0], EXS[0], EXL[0], PB, 0.5, 0.02, 240, False, -9.0, TICK)
b6, bm = np.diff(eq0[B6]), np.diff(eq0[BM]); np.save('/home/claude/zecwork/eq_V29_yatay_baz.npy', eq0)
print(f'baz kontrol (V2.9 = 9061.9): {eq0[-1]:.1f} | hazırlık {time.time() - T0:.0f} sn'); print('yatay süre payı %:', {d[0]: round(d[1].mean() * 100, 1) for d in DET[1:-1]})
CB = [(0, 0, 0, 0, 0, 0, 0)] + [(yi, ei) + t for yi in range(1, len(DET)) for ei in range(len(EX)) for t in itertools.product(range(2), range(3), range(2), range(2), range(3))]
CB = np.array(CB, np.int64); T1 = time.time()
R = grid(CB, PRE, CONT, TBV, GV, o, h, l, c, posA, bXp, bXd, R10, YAT, EXS, EXL, PB, TICK, mid, t23, t300, B6, b6, BM, bm)
print(f'{len(CB)} varyant {time.time() - T1:.0f} sn')
cols = ['net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'mddIS', 'mddOOS', 'Ulcer', 'blok+', 'blok', 'ay+', 'ay-', 'çıkış']
df = pd.DataFrame(R, columns=cols)
df['tespit'] = [DET[k][0] for k in CB[:, 0]]; df['çıkış_tipi'] = [EX[k][0] for k in CB[:, 1]]; df['geriPB'] = PRE[CB[:, 2]]; df['devam'] = CONT[CB[:, 3]]; df['süre'] = TBV[CB[:, 4]]
df['ters'] = CB[:, 5]; df['kârŞartı'] = GV[CB[:, 6]]
b = df.iloc[0]
for k in ('net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'Ulcer'): df['d' + k] = df[k] - b[k]
df.to_pickle('/home/claude/zecwork/yatay.pkl')
