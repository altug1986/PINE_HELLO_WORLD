# TRIM ailesi geniş ızgara: tetik (RSI N/X veya z N/k) × geri giriş (seviye / bar / devam) × trend Sharpe süzgeci × taraf
import sys, time, itertools, numpy as np, pandas as pd, numba
W = int(sys.argv[1]); NW = int(sys.argv[2])
exec(open('/home/claude/zecwork/overlay.py').read().split("JOBS = []")[0].split("W = int(sys.argv[1])")[1].replace('cache=True', 'cache=False') if False else '')
Z = np.load('/home/claude/zecwork/base28.npz')
o, h, l, c, a = Z['o'], Z['h'], Z['l'], Z['c'], Z['a']; posA = Z['posA'].astype(np.int8); tms = Z['tms']; mid = int(Z['mid']); n = len(c); TB = Z['T']; S = pd.Series; TICK = 0.01
eqB = Z['eq']
bEp = np.zeros(n); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
for r in TB:
    ei, xi = int(r[0]), int(r[1]); bEp[ei:xi] = r[3]; bXp[xi] = r[4]; bXd[xi] = int(r[2])
def rsi(x, N):
    d = np.diff(x, prepend=x[0]); u = S(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = S(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
ix = pd.to_datetime(tms, unit='ms'); br = np.r_[0, c[1:] / c[:-1] - 1]; trSh = np.nan_to_num((S(br).rolling(4320).mean() / S(br).rolling(4320).std() * np.sqrt(4320)).to_numpy())
mon = ix.to_period('M'); mcode = (mon.year * 12 + mon.month).to_numpy(); dc = S(c, index=ix).resample('1D').last(); mv = dc.pct_change().abs().groupby(dc.index.to_period('M')).mean()
isCalm = np.isin(mcode, [p.year * 12 + p.month for p in mv[mv <= mv.quantile(1 / 3)].index]); t300 = tms[-1] - 300 * 86400000; yr = ix.year.to_numpy()
base_c = np.zeros(n); np.add.at(base_c, TB[:, 1].astype(int), TB[:, 5])
@numba.njit(cache=True)
def ov2(c, posA, bEp, bXp, bXd, trL, trS, riL, riS, tb, cont, shMax, trSh, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; nt = 0
    eq30 = np.zeros(n // 30 + 1); P = np.zeros(n)
    for i in range(1, n):
        D = posA[i]
        if not np.isnan(bXp[i]) and pos != 0 and pos == bXd[i] and mode == 0:
            xp = bXp[i]; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
        want = D
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if ((D > 0 and trL[i]) or (D < 0 and trS[i])) and trSh[i] * D < shMax:
                    want = 0; mode = 2; tPx = c[i]; tBar = i; tSide = D; nt += 1
            else:
                bi = (riL[i] or i - tBar >= tb or c[i] >= tPx * (1.0 + cont)) if tSide > 0 else (riS[i] or i - tBar >= tb or c[i] <= tPx * (1.0 - cont))
                if bi: mode = 0
                else: want = 0
        prevD = D
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl; P[i] += pnl; pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
        if i % 30 == 0: eq30[i // 30] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq30, P, nt
def metr(lab, eq30, P, nt):
    dd = np.maximum.accumulate(eq30) - eq30; dl = P - base_c
    r = dict(v=lab, net=P.sum(), nTrim=nt, dIS=dl[:mid].sum(), dOOS=dl[mid:].sum(), dCalm=dl[isCalm].sum(), d300=dl[tms >= t300].sum(), MaxDD=dd.max(), DD99=np.percentile(dd, 99), DD95=np.percentile(dd, 95))
    for y in (2021, 2022, 2023, 2024, 2025, 2026): r[f'd{y}'] = dl[yr == y].sum()
    return r
F0 = np.zeros(n, np.bool_); JOBS = []
RS = {N: np.nan_to_num(rsi(c, N), nan=50.0) for N in (7, 10, 14, 21)}
for N, X, Y, tb, cont, sh in itertools.product((7, 10, 14, 21), (86, 88, 90, 92, 94), (40, 50, 60, 70), (12, 60, 240, 720), (0.01, 0.02, 0.03, 0.05), (1.5, 2.0, 3.0, 99.0)):
    JOBS.append((f'RSI{N} X{X} Y{Y} geri{tb} devam%{cont * 100:g} sh<{sh}', ('r', N, X, Y, 'iki'), tb, cont, sh))
for X, Y, tb, cont, sh, side in itertools.product((86, 88, 90, 92, 94), (40, 50, 60, 70), (12, 60, 240, 720), (0.01, 0.02, 0.03, 0.05), (2.0, 99.0), ('long', 'short')):
    JOBS.append((f'RSI14 X{X} Y{Y} geri{tb} devam%{cont * 100:g} sh<{sh} yalnız {side}', ('r', 14, X, Y, side), tb, cont, sh))
ZZ = {}
for Nz in (30, 60, 120):
    s = S(c).rolling(Nz); ZZ[Nz] = np.nan_to_num(((S(c) - s.mean()) / s.std()).to_numpy())
for Nz, k, y, tb, cont, sh in itertools.product((30, 60, 120), (2.5, 3.0, 3.5, 4.0), (0.0, 1.0, 1.5), (12, 60, 240, 720), (0.01, 0.02, 0.03, 0.05), (2.0, 99.0)):
    JOBS.append((f'Z{Nz} k{k} geri z≤{y} geri{tb} devam%{cont * 100:g} sh<{sh}', ('z', Nz, k, y, 'iki'), tb, cont, sh))
print(f'[{W}] iş {len(JOBS)}', flush=True); rows = []; t0 = time.time(); cache = {}
for j, (lab, tg, tb, cont, sh) in enumerate(JOBS):
    if j % NW != W: continue
    if tg not in cache:
        cache.clear()
        if tg[0] == 'r':
            R_ = RS[tg[1]]; trL, trS, riL, riS = R_ >= tg[2], R_ <= 100 - tg[2], R_ <= tg[3], R_ >= 100 - tg[3]
        else:
            Zz = ZZ[tg[1]]; trL, trS, riL, riS = Zz >= tg[2], Zz <= -tg[2], Zz <= tg[3], Zz >= -tg[3]
        if tg[4] == 'long': trS = F0
        if tg[4] == 'short': trL = F0
        cache[tg] = (trL, trS, riL, riS)
    trL, trS, riL, riS = cache[tg]
    eq30, P, nt = ov2(c, posA, bEp, bXp, bXd, trL, trS, riL, riS, tb, cont, sh, trSh, TICK); rows.append(metr(lab, eq30, P, nt))
    if j % 500 == W: print(f'[{W}] {j}/{len(JOBS)} {time.time() - t0:.0f} sn', flush=True)
pd.DataFrame(rows).to_pickle(f'/home/claude/zecwork/ov2_{W}.pkl'); print(f'[{W}] bitti {len(rows)} {time.time() - t0:.0f} sn', flush=True)
