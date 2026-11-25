# DEVASA ADAY İŞLEM + FİLTRE (meta-etiketleme araştırması)
# V2.9'un 2.502 işleminin 1/10/20/50/100/200 katı aday işlem üret (her k barda long+short, H bar tut, maliyet dahil),
# özellikler (nedensel) → yarıda eğit / öbür yarıda test (arınma H bar), sonra V2.9'a uygula: (A) giriş vetosu, (B) baz FLAT iken dolgu.
src = open('/home/claude/zecwork/pffilt.py').read(); exec(src.split("rows = [ev(")[0])
from pfsearch import bEp
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
T0 = time.time(); f32 = np.float32
def rma(x, N): return S(x).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
def lagd(x, k): return x - np.r_[np.full(k, np.nan), x[:-k]]
lc = np.log(c); FT = {}
for k in (15, 60, 240, 1440): FT[f'r{k}'] = lagd(lc, k)
for N in (60, 240, 1440): FT[f'z{N}'] = (c - S(c).rolling(N).mean().values) / S(c).rolling(N).std().values
FT['rsi10'] = R10 - 50; FT['rsi60'] = P.rsi(c, 60) - 50; FT['rsi240'] = P.rsi(c, 240) - 50
btc = S(Z['btc'].astype(float)).ffill().values; xmr = S(Z['xmr'].astype(float)).ffill().values
for k in (15, 60, 240, 1440): FT[f'b{k}'] = lagd(np.log(btc), k)
FT['rel240'] = FT['r240'] - FT['b240']; FT['rel1440'] = FT['r1440'] - FT['b1440']; FT['x240'] = lagd(np.log(xmr), 240); FT['x1440'] = lagd(np.log(xmr), 1440)
FT['fund'] = S(Z['fund'].astype(float)).ffill().values; FT['basis'] = S(Z['basis'].astype(float)).ffill().values; FT['stt'] = Z['stt'].astype(float); FT['baz'] = posA.astype(float)
DIRF = list(FT.keys())                                   # yönlü özellikler: aday yönüyle çarpılır (long/short ayna)
pc = np.r_[c[0], c[:-1]]; trr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))); FT['atrp'] = rma(trr, 14) / c
dl = np.r_[0, np.diff(lc)]; FT['volreg'] = S(dl).rolling(1440).std().values / S(dl).rolling(43200, min_periods=10000).std().values
vol = Z['vol'].astype(float); FT['vr'] = vol / S(vol).rolling(1440, min_periods=60).mean().shift(1).values
oi = S(Z['oi'].astype(float)).ffill().values
for k in (15, 60, 240, 1440): FT[f'oi{k}'] = lagd(np.log(oi), k)
def htf(rule): return pd.DataFrame({'h': h, 'l': l, 'c': c}, ix).resample(rule).agg({'h': 'max', 'l': 'min', 'c': 'last'}).dropna()
def mapH(v, D, td): s_ = S(v, D.index + td); j = np.searchsorted(s_.index.values, ix.values, side='right') - 1; out = s_.values[np.maximum(j, 0)]; out[j < 0] = np.nan; return out
def trng(D): H_, L_, C_ = D.h.values, D.l.values, D.c.values; p_ = np.r_[C_[0], C_[:-1]]; return H_, L_, np.maximum(H_ - L_, np.maximum(np.abs(H_ - p_), np.abs(L_ - p_)))
D4, D1 = htf('4h'), htf('1h'); H4, L4, t4 = trng(D4); up = np.r_[0, np.diff(H4)]; dn = np.r_[0, -np.diff(L4)]; a4 = rma(t4, 14)
pdi = 100 * rma(np.where((up > dn) & (up > 0), up, 0.0), 14) / a4; mdi = 100 * rma(np.where((dn > up) & (dn > 0), dn, 0.0), 14) / a4
FT['adx4'] = mapH(rma(100 * np.abs(pdi - mdi) / np.where(pdi + mdi == 0, 1, pdi + mdi), 14), D4, pd.Timedelta('4h'))
H1_, L1_, t1 = trng(D1); FT['chop1'] = mapH(100 * np.log10(S(t1).rolling(14).sum().values / np.maximum(S(H1_).rolling(14).max().values - S(L1_).rolling(14).min().values, 1e-12)) / np.log10(14), D1, pd.Timedelta('1h'))
bage = np.zeros(n)
for r in TB: bage[int(r[0]):int(r[1])] = np.arange(int(r[1]) - int(r[0]))
FT['bage'] = bage; FT['bpnl'] = np.where(posA != 0, (c / np.where(bEp > 0, bEp, c) - 1) * posA, 0.0)
NAMES = list(FT.keys()); ND = len(DIRF); F = np.column_stack([FT[k].astype(f32) for k in NAMES]); del FT
print(f'özellik {len(NAMES)} | hazırlık {time.time() - T0:.0f} sn', flush=True)
def cand(bars, H):
    X = np.repeat(F[bars], 2, axis=0); d = np.tile(np.array([1, -1]), len(bars)); X[:, :ND] *= d[:, None].astype(f32); X = np.column_stack([X, d.astype(f32)])
    bi = np.repeat(bars, 2); en = c[bi] + TICK * d; ex = c[bi + H] - TICK * d; q = 500.0 / c[bi]
    return X, q * (ex - en) * d - 0.0004 * q * (en + ex), bi, d
def Xat(bars, d):
    X = F[bars].copy(); X[:, :ND] *= f32(d); return np.column_stack([X, np.full(len(bars), d, f32)])
def Xtr(bars, dv):
    X = F[bars].copy(); X[:, :ND] *= dv[:, None].astype(f32); return np.column_stack([X, dv.astype(f32)])
def fit(X, y):
    lo, hi = np.percentile(y, [1, 99]); m = HGB(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=max(100, len(y) // 100), l2_regularization=1.0, early_stopping=False)
    return m.fit(X, np.clip(y, lo, hi))
W0 = 2000; KS = (1, 10, 20, 50, 100, 200); HS = (240, 960); BASE_N = K
wk = ((tms - tms[0]) // (7 * 86400000)).astype(np.int64)
xb = np.flatnonzero(vpos == 0); fl = vpos == 0
@njit(cache=False)
def followB(o, c, tick, vpos, vxp, sig, H):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; mb = 0; isB = False; eq = np.zeros(n); nb = 0; bp = 0.0
    for i in range(1, n):
        if real != 0 and not isB and real == vpos[i - 1] and vxp[i] > 0:
            rz += q * (vxp[i] - ep) * real - 0.0004 * q * (ep + vxp[i]); real = 0
        v = vpos[i]
        if v != 0: tg = v
        elif isB and i - mb < H: tg = real
        else:
            tg = sig[i]
            if isB and tg == real and tg != 0: mb = i
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn
                if isB: bp += pn
                real = 0; isB = False
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if v == 0: isB = True; mb = i; nb += 1
        elif v != 0: isB = False
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nb, bp
out = []; EV = {}
for H in HS:
    sA = lambda b: b <= mid - H; sB = lambda b: b >= mid + H
    dense = np.arange(W0, n - H - 1, max(1, int((n - W0 - H) / (100 * BASE_N / 2))))     # ortak test kümesi (100x)
    Xd, yd, bd, dd = cand(dense, H)
    for k in KS:
        st = max(1, int((n - W0 - H) / (k * BASE_N / 2))); bars = np.arange(W0 + (k * 7) % st, n - H - 1, st); X, y, bi, d = cand(bars, H)
        res = dict(çarpan=k, H=H, aday=len(y), etkinÖrnek=int((n - 2 * H) / H))
        mods = {}
        for nm, trn, tst in (('A', sA, sB), ('B', sB, sA)):
            m = fit(X[trn(bi)], y[trn(bi)]); mods[nm] = m; te = tst(bd); p = m.predict(Xd[te]); yy = yd[te]
            res[f'ρ_{nm}'] = pd.Series(p).corr(pd.Series(yy), method='spearman'); top = p >= np.quantile(p, 0.9)
            res[f'top10_{nm}'] = yy[top].mean() - yy.mean(); w = pd.DataFrame({'w': wk[bd[te]], 'd': np.where(top, yy - yy.mean(), np.nan)}).groupby('w').d.mean().dropna()
            res[f't_{nm}'] = w.mean() / (w.std() / np.sqrt(len(w)))
            mods[nm + 'cq'] = {x: np.quantile(m.predict(X[trn(bi)]), 1 - x) for x in (0.02, 0.05, 0.10)}
        # (A) V2.9 giriş vetosu — çapraz: ilk yarı işlemleri B modeliyle, ikinci yarı A modeliyle (A = geçmişten öğrenen)
        pv = np.empty(K); pin = np.empty(K)
        for nm, sel in (('B', ei < mid), ('A', ei >= mid)):
            pv[sel] = mods[nm].predict(Xtr(ei[sel], TR[sel, 2])); pin[~sel] = mods[nm].predict(Xtr(ei[~sel], TR[~sel, 2]))  # eşik: modelin kendi eğitim yarısındaki V2.9 işlemleri
        for qq in (0.1, 0.2, 0.3):
            keep = np.ones(K, bool)
            for nm, sel in (('B', ei < mid), ('A', ei >= mid)):
                thr = np.quantile(pin[~sel], qq); keep[sel] = pv[sel] >= thr
            Fv = np.ones(n, np.bool_); Fv[ei[~keep]] = False; e_, nt_, pf_, im_ = follow(o, c, TICK, vpos, vxp, vnew, Fv, 3)
            res[f'veto%{int(qq * 100)}_dNet'] = e_[-1] - eq0[-1]; res[f'veto%{int(qq * 100)}_dIS'] = e_[mid] - eq0[mid]; res[f'veto%{int(qq * 100)}_dOOS'] = (e_[-1] - e_[mid]) - (eq0[-1] - eq0[mid])
            res[f'veto%{int(qq * 100)}_vetoNet'] = pnl[~keep].sum(); x30 = e_[::30]; res[f'veto%{int(qq * 100)}_MaxDD'] = (np.maximum.accumulate(x30) - x30).max()
        # (B) baz FLAT iken dolgu: tahmin ≥ eğitim adaylarının üst %x eşiği → H bar tut
        PLS = {nm: (xb[sel], mods[nm].predict(Xat(xb[sel], 1)), mods[nm].predict(Xat(xb[sel], -1))) for nm, sel in (('B', xb < mid), ('A', xb >= mid))}
        for x in (0.02, 0.05, 0.10):
            sig = np.zeros(n, np.int64)
            for nm in ('B', 'A'):
                bb, pL, pS = PLS[nm]; th = mods[nm + 'cq'][x]
                sig[bb] = np.where((pL >= th) & (pL >= pS), 1, np.where((pS >= th) & (pS > pL), -1, 0))
            e_, nb_, bp_ = followB(o, c, TICK, vpos, vxp, sig, H); x30 = e_[::30]
            res[f'dolgu%{int(x * 100)}_dNet'] = e_[-1] - eq0[-1]; res[f'dolgu%{int(x * 100)}_dIS'] = e_[mid] - eq0[mid]; res[f'dolgu%{int(x * 100)}_dOOS'] = (e_[-1] - e_[mid]) - (eq0[-1] - eq0[mid])
            res[f'dolgu%{int(x * 100)}_adet'] = nb_; res[f'dolgu%{int(x * 100)}_MaxDD'] = (np.maximum.accumulate(x30) - x30).max()
        out.append(res); print({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in res.items()}, f'{time.time() - T0:.0f} sn', flush=True)
pd.DataFrame(out).to_pickle('/home/claude/zecwork/buyuk.pkl')
