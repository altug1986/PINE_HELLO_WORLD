# TREND MOTORU: 4 s Donchian kırılım giriş + kanal / ATR iz çıkış; sabit 500 $ giriş miktarı trend boyunca taşınır. V2.7 ile tek birim (trend öncelikli)
exec(open('/home/claude/zecwork/vbar1.py').read().split("eq, T = sim(X0); ad('V2.7 baz', eq, T)")[0])
importlib.reload(z03)
Z4 = pd.DataFrame({'o': o, 'h': h, 'l': l, 'c': c}, idx).resample('4h').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
tr4 = pd.concat([Z4.h - Z4.l, (Z4.h - Z4.c.shift()).abs(), (Z4.l - Z4.c.shift()).abs()], axis=1).max(axis=1); A4 = tr4.ewm(alpha=1 / 14, adjust=False).mean().values
def state(N, M, k, both):
    hi = Z4.h.rolling(N).max().shift(1).values; lo = Z4.l.rolling(N).min().shift(1).values; xl = Z4.l.rolling(M).min().shift(1).values; xh = Z4.h.rolling(M).max().shift(1).values
    C = Z4.c.values; st_ = np.zeros(len(C)); s = 0; ext = 0.0
    for t in range(len(C)):
        if s == 1:
            ext = max(ext, C[t])
            if C[t] < xl[t] or (k > 0 and C[t] < ext - k * A4[t]): s = 0
        elif s == -1:
            ext = min(ext, C[t])
            if C[t] > xh[t] or (k > 0 and C[t] > ext + k * A4[t]): s = 0
        if s == 0:
            if C[t] > hi[t]: s = 1; ext = C[t]
            elif both and C[t] < lo[t]: s = -1; ext = C[t]
        st_[t] = s
    return np.nan_to_num(map1m(pd.Series(st_, Z4.index), 4))
def simT(stt, withOrk=True):
    z = np.zeros(n, np.bool_); tl = stt > 0; ts = stt < 0
    rows = [(z, z, z, z), (tl, ts, tl, ts)] + ([tuple(X0[q][r] for q in range(4)) for r in range(1, 6)] if withOrk else [])
    X = [np.ascontiguousarray(np.vstack([r_[q] for r_ in rows])) for q in range(4)]
    TS_ = np.r_[1e6, 1e6, TS6[1:]] if withOrk else np.r_[1e6, 1e6]; TH_ = np.r_[480, 1, TH6[1:]] if withOrk else np.r_[480, 1]
    by = np.r_[0, 1, byp[1:]].astype(np.bool_) if withOrk else np.array([0, 1], np.bool_); cA = np.r_[0.0, 0.0, [0.10] * 5] if withOrk else np.r_[0.0, 0.0]
    return z03.simN11(o, h, l, c, a, *X, TS_, TH_, spk, 60, 3, by, 0.10, False, F0, F0, cA)
eq, T = sim(X0); ad('V2.7 baz', eq, T)
for N, M in ((20, 10), (55, 20), (100, 50)):
    for k in (0, 4, 8):
        for both in (False, True):
            stt = state(N, M, k, both); nm = f'Donchian {N}/{M} iz{k}×ATR4s {"iki yön" if both else "yalnız long"}'
            eq, T = simT(stt, False); r = ad('TEK: ' + nm, eq, T)
            eq, T = simT(stt, True); ad('V2.7+' + nm, eq, T); out[-1]['trend$'] = T[T[:, 7] == 1, 5].sum(); out[-1]['trendN'] = int((T[:, 7] == 1).sum())
            np.save(f'/home/claude/zecwork/eq_TR_{N}_{k}_{int(both)}.npy', eq[::30]); np.save(f'/home/claude/zecwork/T_TR_{N}_{k}_{int(both)}.npy', T)
pd.DataFrame(out).to_csv('/home/claude/pine_hello_world/ZEC/research/TREND_MOTORU_1DK.csv', index=False)
