# POZİSYON İÇİ KATMAN (PEPE V11 mantığı V2.8 yönü üstünde): mikro kâr al / geri gir, karşı pozisyon, OI tükeniş kilidi.
import sys, time, itertools, numpy as np, pandas as pd, numba
W = int(sys.argv[1]); NW = int(sys.argv[2])
Z = np.load('/home/claude/zecwork/base28.npz')
o, h, l, c, a = Z['o'], Z['h'], Z['l'], Z['c'], Z['a']; posA = Z['posA'].astype(np.int8); tms = Z['tms']; mid = int(Z['mid']); n = len(c); TB = Z['T']; S = pd.Series; TICK = 0.01
eqB = Z['eq']; eq30 = eqB[::30]
bEp = np.zeros(n); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
for r in TB:
    ei, xi = int(r[0]), int(r[1]); bEp[ei:xi] = r[3]; bXp[xi] = r[4]; bXd[xi] = int(r[2])
def L(x, k): return np.r_[np.full(k, np.nan), x[:-k]]
def rsi(x, N):
    d = np.diff(x, prepend=x[0]); u = S(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = S(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
ix = pd.to_datetime(tms, unit='ms'); H1 = pd.DataFrame({'h': h, 'l': l, 'c': c}, ix).resample('1h').agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
tr = np.maximum(H1.h - H1.l, np.maximum((H1.h - H1.c.shift()).abs(), (H1.l - H1.c.shift()).abs())); atrH = (tr.ewm(alpha=1 / 14, adjust=False).mean() / H1.c * 100).shift(1)
atr1h = atrH.reindex(ix.floor('1h')).to_numpy()
br = np.r_[0, c[1:] / c[:-1] - 1]; trSh = np.nan_to_num((S(br).rolling(4320).mean() / S(br).rolling(4320).std() * np.sqrt(4320)).to_numpy())
oi = S(Z['oi']).ffill().to_numpy()
RS = {N: np.nan_to_num(rsi(c, N), nan=50.0) for N in (14, 35, 70)}
DP = {N: (c / L(c, N) - 1) * 100 for N in (30, 60, 240)}; DO = {N: (oi / L(oi, N) - 1) * 100 for N in (30, 60, 240)}
mon = pd.to_datetime(tms, unit='ms').to_period('M'); mcode = (mon.year * 12 + mon.month).to_numpy()
dc = S(c, index=ix).resample('1D').last(); mv = dc.pct_change().abs().groupby(dc.index.to_period('M')).mean(); calmM = [p.year * 12 + p.month for p in mv[mv <= mv.quantile(1 / 3)].index]
isCalm = np.isin(mcode, calmM); t300 = tms[-1] - 300 * 86400000; yr = ix.year.to_numpy()
@numba.njit(cache=True)
def ov(o, h, l, c, posA, bEp, bXp, bXd, rsiT, X, Y, tb, cont, shMax, trSh, kC, atr1h, cSL, cB, exL, exS, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; cDir = 0; cPx = 0.0; cBar = 0
    eq = np.zeros(n); P = np.zeros(n)
    for i in range(1, n):
        D = posA[i]
        # baz bu barda stop/çıkışla kapandıysa ve biz aynı yöndeysek aynı fiyattan çık
        if not np.isnan(bXp[i]) and pos != 0 and pos == bXd[i] and mode == 0:
            xp = bXp[i]; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
        want = D
        if D != prevD:
            mode = 0
        elif D != 0:
            if mode == 0:
                adv = (c[i] / bEp[i] - 1.0) * (-D) * 100.0
                if kC > 0 and not np.isnan(atr1h[i]) and adv >= kC * atr1h[i]:
                    want = -D; mode = 1; cDir = D; cPx = c[i]; cBar = i
                elif (D > 0 and exL[i]) or (D < 0 and exS[i]) or (X > 0 and ((D > 0 and rsiT[i] >= X) or (D < 0 and rsiT[i] <= 100.0 - X)) and trSh[i] * D < shMax):
                    want = 0; mode = 2; tPx = c[i]; tBar = i; tSide = D
            elif mode == 1:
                g = (c[i] / cPx - 1.0) * (-cDir)
                if g <= -cSL or i - cBar >= cB: mode = 0
                else: want = -cDir
            else:
                bi = (rsiT[i] <= Y or i - tBar >= tb or c[i] >= tPx * (1.0 + cont)) if tSide > 0 else (rsiT[i] >= 100.0 - Y or i - tBar >= tb or c[i] <= tPx * (1.0 - cont))
                if bi: mode = 0
                else: want = 0
        prevD = D
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, P
F0 = np.zeros(n, np.bool_)
def metr(lab, eq, P):
    e30 = eq[::30]; dd = np.maximum.accumulate(e30) - e30; d = P - np.r_[0, np.diff(eqB)] * 0          # P: kapanan kâr barları
    cumP = np.cumsum(P); base_c = np.zeros(n); np.add.at(base_c, TB[:, 1].astype(int), TB[:, 5]); cB_ = np.cumsum(base_c)
    dl = P - base_c                                                                                 # bar bazında gerçekleşen fark
    r = dict(v=lab, net=eq[-1], dNet=eq[-1] - eqB[-1], dIS=dl[:mid].sum(), dOOS=dl[mid:].sum(), dCalm=dl[isCalm].sum(), d300=dl[tms >= t300].sum(),
             MaxDD=dd.max(), DD99=np.percentile(dd, 99), DD95=np.percentile(dd, 95))
    for y in (2021, 2022, 2023, 2024, 2025, 2026): r[f'd{y}'] = dl[yr == y].sum()
    x = np.diff(e30[::48]); r['Sh'] = x.mean() / x.std() * np.sqrt(365); return r
JOBS = []
for N, X, Y, tb, cont, sh in itertools.product((14, 35, 70), (80, 85, 90, 95), (50, 60, 70), (12, 60, 240), (0.01, 0.03), (2.0, 99.0)):
    JOBS.append((f'TRIM rsi{N} X{X} Y{Y} geri{tb} devam%{cont * 100:g} sh<{sh}', dict(rsiT=RS[N], X=float(X), Y=float(Y), tb=tb, cont=cont, shMax=sh, kC=0.0, cSL=0.02, cB=144, exL=F0, exS=F0)))
for kC, cSL, cB in itertools.product((1.5, 2.0, 3.0, 4.5, 6.0, 8.0), (0.01, 0.02, 0.03, 0.05), (60, 240, 720, 1440)):
    JOBS.append((f'KARŞI k{kC}×ATR1s sl%{cSL * 100:g} süre{cB}', dict(rsiT=RS[14], X=0.0, Y=60.0, tb=60, cont=0.03, shMax=99.0, kC=kC, cSL=cSL, cB=cB, exL=F0, exS=F0)))
for N, px, oz, Y, tb in itertools.product((30, 60, 240), (1.0, 2.0, 3.0, 5.0), (1.0, 2.0, 3.0), (50, 60), (60, 240)):
    exL_ = np.nan_to_num((DP[N] >= px) & (DO[N] <= -oz)).astype(np.bool_); exS_ = np.nan_to_num((DP[N] <= -px) & (DO[N] <= -oz)).astype(np.bool_)
    JOBS.append((f'TÜKENİŞ {N}dk fiyat%{px} OI−%{oz} Y{Y} geri{tb}', dict(rsiT=RS[14], X=0.0, Y=float(Y), tb=tb, cont=0.03, shMax=99.0, kC=0.0, cSL=0.02, cB=144, exL=exL_, exS=exS_)))
print(f'[{W}] iş {len(JOBS)}', flush=True); rows = []; t0 = time.time()
eq, P = ov(o, h, l, c, posA, bEp, bXp, bXd, RS[14], 0.0, 60.0, 60, 0.03, 99.0, trSh, 0.0, atr1h, 0.02, 144, F0, F0, TICK)
if W == 0: print('kontrol: katmansız =', round(eq[-1], 1), 'baz', round(eqB[-1], 1), flush=True)
for j, (lab, kw) in enumerate(JOBS):
    if j % NW != W: continue
    eq, P = ov(o, h, l, c, posA, bEp, bXp, bXd, kw['rsiT'], kw['X'], kw['Y'], kw['tb'], kw['cont'], kw['shMax'], trSh, kw['kC'], atr1h, kw['cSL'], kw['cB'], kw['exL'], kw['exS'], TICK)
    rows.append(metr(lab, eq, P))
pd.DataFrame(rows).to_pickle(f'/home/claude/zecwork/ov_{W}.pkl'); print(f'[{W}] bitti {len(rows)} {time.time() - t0:.0f} sn', flush=True)
