exec(open('/home/claude/zecwork/overlay3.py').read().split("F0 = np.zeros(n, np.bool_); RS = ")[0])
from scipy.stats import norm, skew, kurtosis
@numba.njit
def ov4(o, h, l, c, posA, bEp, bXp, bXd, trL, trS, riL, riS, tb, cont, shMax, trSh, flip, fSL, tick, fee, nextOpen):
    """ov3 + icra seçeneği: katmanın KENDİ emirleri (kâr al / ters / geri gir) bir sonraki barın açılışında dolar (nextOpen=1); baz takibi aynı."""
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; nt = 0; st = 0.0; pend = 0; pendSet = False
    eq = np.zeros(n)
    for i in range(1, n):
        if pendSet:                                   # önceki barın katman emri bu barın açılışında
            want_ = pend; pendSet = False
            if want_ != pos:
                if pos != 0:
                    xp = o[i] - tick * pos; real += q * (xp - ep) * pos - fee * q * (ep + xp); pos = 0
                if want_ != 0:
                    pos = want_; ep = o[i] + tick * pos; q = 500.0 / o[i]; st = ep * (1.0 - pos * fSL)
        D = posA[i]
        if not np.isnan(bXp[i]) and pos != 0 and pos == bXd[i] and mode == 0:
            xp = bXp[i]; real += q * (xp - ep) * pos - fee * q * (ep + xp); pos = 0
        if mode == 2 and flip and pos != 0 and pos == -tSide and ((pos == 1 and l[i] <= st) or (pos == -1 and h[i] >= st)):
            xp = (min(o[i], st) - tick) if pos == 1 else (max(o[i], st) + tick); real += q * (xp - ep) * pos - fee * q * (ep + xp); pos = 0; mode = 0
        want = D; ovr = False
        if D != prevD: mode = 0
        elif D != 0:
            if mode == 0:
                if ((D > 0 and trL[i]) or (D < 0 and trS[i])) and trSh[i] * D < shMax:
                    want = -D if flip else 0; mode = 2; tPx = c[i]; tBar = i; tSide = D; nt += 1; ovr = True
            else:
                bi = (riL[i] or i - tBar >= tb or c[i] >= tPx * (1.0 + cont)) if tSide > 0 else (riS[i] or i - tBar >= tb or c[i] <= tPx * (1.0 - cont))
                if bi: mode = 0; ovr = True
                else: want = (-tSide if pos == -tSide else 0) if flip else 0
        prevD = D
        if want != pos:
            if nextOpen and ovr and i + 1 < n:
                pend = want; pendSet = True
            else:
                if pos != 0:
                    xp = c[i] - tick * pos; real += q * (xp - ep) * pos - fee * q * (ep + xp); pos = 0
                if want != 0:
                    pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]; st = ep * (1.0 - pos * fSL)
        eq[i] = real + (q * (c[i] - ep) * pos - fee * q * ep if pos != 0 else 0.0)
    return eq, nt
F0 = np.zeros(n, np.bool_); R10 = np.nan_to_num(rsi(c, 10), nan=50.0); out = []
def mm(e):
    e30 = e[::30]; dd = np.maximum.accumulate(e30) - e30; return e[-1], dd.max(), np.percentile(dd, 99), e[mid]
for lab, fee, tk, nxt in (('normal (kapanışta dolum)', 0.0004, 0.01, 0), ('KATMAN emirleri sonraki açılışta', 0.0004, 0.01, 1), ('stres %0.08 + 3 tick', 0.0008, 0.03, 0), ('stres + sonraki açılış', 0.0008, 0.03, 1)):
    e0, _ = ov4(o, h, l, c, posA, bEp, bXp, bXd, F0, F0, F0, F0, 240, 0.03, 99.0, trSh, True, 0.02, tk, fee, nxt)
    e1, nt = ov4(o, h, l, c, posA, bEp, bXp, bXd, R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, 240, 0.03, 99.0, trSh, True, 0.02, tk, fee, nxt)
    a0, a1 = mm(e0), mm(e1); out.append(dict(icra=lab, kontrol=a0[0], seçilen=a1[0], fark=a1[0] - a0[0], farkIS=a1[3] - a0[3], farkOOS=(a1[0] - a1[3]) - (a0[0] - a0[3]), MaxDD=f'{a0[1]:.0f}→{a1[1]:.0f}', DD99=f'{a0[2]:.0f}→{a1[2]:.0f}', n=nt))
pd.set_option('display.width', 250); print(pd.DataFrame(out).round(1).to_string(index=False))
# DSR tüm strateji (birleşik günlük getiri), denemeler: TRIM ızgarası birleşik SR dağılımı
ix_ = pd.to_datetime(tms, unit='ms')
e0, _ = ov4(o, h, l, c, posA, bEp, bXp, bXd, F0, F0, F0, F0, 240, 0.03, 99.0, trSh, True, 0.02, TICK, 0.0004, 0)
e1, _ = ov4(o, h, l, c, posA, bEp, bXp, bXd, R10 >= 94, R10 <= 6, R10 <= 70, R10 >= 30, 240, 0.03, 99.0, trSh, True, 0.02, TICK, 0.0004, 0)
d1 = pd.Series(e1, ix_).resample('1D').last().diff().dropna().to_numpy(); d0 = pd.Series(e0, ix_).resample('1D').last().diff().dropna().to_numpy()
import re; R2 = pd.read_pickle('/home/claude/zecwork/ov2_all.pkl'); RS2 = {N: np.nan_to_num(rsi(c, N), nan=50.0) for N in (7, 10, 14, 21)}; srs = []
for v in R2.sample(300, random_state=2).v:
    m_ = re.match(r'RSI(\d+) X(\d+) Y(\d+) geri(\d+) devam%([\d.]+) sh<([\d.]+)', v)
    if not m_: continue
    Rr = RS2[int(m_[1])]; X, Y = float(m_[2]), float(m_[3])
    e, _ = ov4(o, h, l, c, posA, bEp, bXp, bXd, Rr >= X, Rr <= 100 - X, Rr <= Y, Rr >= 100 - Y, int(m_[4]), float(m_[5]) / 100, float(m_[6]), trSh, False, 0.02, TICK, 0.0004, 0)
    x = pd.Series(e, ix_).resample('1D').last().diff().dropna().to_numpy(); srs.append(x.mean() / x.std())
srs = np.array(srs); NT = 29194; emc = 0.5772156649; SR = d1.mean() / d1.std(); SRb = d0.mean() / d0.std()
SR0 = np.sqrt(srs.var()) * ((1 - emc) * norm.ppf(1 - 1 / NT) + emc * norm.ppf(1 - 1 / (NT * np.e))); g3 = skew(d1); g4 = kurtosis(d1, fisher=False)
z = lambda s0: (SR - s0) * np.sqrt(len(d1) - 1) / np.sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2)
print(f'TÜM STRATEJİ günlük SR {SR:.4f} (yıllık {SR * np.sqrt(365):.2f}) · baz {SRb:.4f} ({SRb * np.sqrt(365):.2f}) · denemeler arası SR std {srs.std():.4f} · SR0 {SR0:.4f}')
print(f'DSR (beklenen en iyi deneme SR0 karşısında) = {norm.cdf(z(SR0)):.3f} · PSR (SR > baz SR) = {norm.cdf(z(SRb)):.3f} · PSR (SR > 0) = {norm.cdf(z(0)):.3f}')
