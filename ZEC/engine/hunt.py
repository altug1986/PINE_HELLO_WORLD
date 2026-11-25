# CYBORG ZEC — FLAT-DOLGU AVI: V2.8 flat iken çalışan aday motorlar (12 aile, ~12–18 bin varyant).
# Aday yalnız baz flat iken girer; baz girince (aynı bar kapanışı) çıkar. Seçim ilk yarı (IS), test ikinci yarı (OOS).
import sys, time, itertools, numpy as np, pandas as pd, numba
W = int(sys.argv[1]); NW = int(sys.argv[2])            # işçi no / işçi sayısı
Z = np.load('/home/claude/zecwork/base28.npz')
o, h, l, c, a = Z['o'], Z['h'], Z['l'], Z['c'], Z['a']; posA = Z['posA'].astype(np.int8); tms = Z['tms']; mid = int(Z['mid']); n = len(c)
eq30 = Z['eq'][::30]; S = pd.Series; TICK = 0.01
def L(x, k): return np.r_[np.full(k, np.nan), x[:-k]]
def ret(x, k): return x / L(x, k) - 1
def zroll(x, N, minp=None): s = S(x); r = s.rolling(N, min_periods=minp or N); return ((s - r.mean()) / r.std()).to_numpy()
t0 = time.time()
# ── ortak özellikler ──
atrp = a / c; calm = atrp < 0.8 * S(atrp).rolling(43200, min_periods=10000).mean().to_numpy(); hot = ~calm
up = c > S(c).ewm(span=48000, adjust=False).mean().to_numpy(); REG = {'hepsi': None, 'sakin': calm, 'sakin değil': hot}
ZN = {N: zroll(c, N) for N in (30, 60, 120, 240, 480, 1440)}
def rsi(x, N):
    d = np.diff(x, prepend=x[0]); u = S(np.maximum(d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy(); v = S(np.maximum(-d, 0)).ewm(alpha=1 / N, adjust=False).mean().to_numpy()
    return 100 - 100 / (1 + u / np.where(v == 0, 1e-12, v))
RS = {N: rsi(c, N) for N in (14, 70)}
DH = {N: L(S(h).rolling(N).max().to_numpy(), 1) for N in (60, 240, 720, 1440)}; DL = {N: L(S(l).rolling(N).min().to_numpy(), 1) for N in (60, 240, 720, 1440)}
btc = S(Z['btc']).ffill().to_numpy(); xmr = S(Z['xmr']).ffill().to_numpy()
BR = {k: ret(btc, k) for k in (5, 15, 30, 60)}; XR = {k: ret(xmr, k) for k in (5, 15, 30, 60)}; ZR = {k: ret(c, k) for k in (5, 15, 30, 60, 240)}
BZ = {k: BR[k] / S(BR[k]).rolling(43200, min_periods=10000).std().to_numpy() for k in BR}; XZ = {k: XR[k] / S(XR[k]).rolling(43200, min_periods=10000).std().to_numpy() for k in XR}
basis = S(Z['basis']).ffill().to_numpy(); BS = {N: zroll(basis, N, N // 2) for N in (240, 1440, 4320)}
oi = S(Z['oi']).ffill().to_numpy(); OZ = {k: (lambda r_: r_ / S(r_).rolling(43200, min_periods=10000).std().to_numpy())(ret(oi, k)) for k in (15, 60, 240)}
vol = np.nan_to_num(Z['vol']); VZ = vol / np.maximum(S(vol).rolling(1440, min_periods=500).mean().to_numpy(), 1e-9)
fund = S(Z['fund']).ffill().to_numpy(); mins = (tms // 60000) % 480                  # 8 saatlik funding döngüsünde dakika (00/08/16 UTC = 0)
hr = (tms // 3600000) % 24
print(f'[{W}] özellikler {time.time() - t0:.0f} sn', flush=True)
# ── değerlendirme ──
mon = pd.to_datetime(tms, unit='ms').to_period('M'); mcode = (mon.year * 12 + mon.month).to_numpy()
dc = S(c, index=pd.to_datetime(tms, unit='ms')).resample('1D').last(); mv = dc.pct_change().abs().groupby(dc.index.to_period('M')).mean()
calmM = set((p.year * 12 + p.month) for p in mv[mv <= mv.quantile(1 / 3)].index)
isCalmBar = np.isin(mcode, list(calmM)); t300 = tms[-1] - 300 * 86400000
@numba.njit(cache=True)
def fill(o, h, l, c, posA, sig, exL, exS, H, sl, cd, tick):
    n = len(c); pos = 0; ep = 0.0; q = 0.0; ei = 0; st = 0.0; real = 0.0; last = -10 ** 9
    add30 = np.zeros(n // 30 + 1); T = np.zeros((n // 4, 4)); nt = 0
    for i in range(1, n):
        if pos != 0:
            xp = 0.0; ex = False
            if (pos == 1 and l[i] <= st) or (pos == -1 and h[i] >= st):
                xp = (min(o[i], st) - tick) if pos == 1 else (max(o[i], st) + tick); ex = True
            elif posA[i] != 0 or i - ei >= H or (pos == 1 and exL[i]) or (pos == -1 and exS[i]):
                xp = c[i] - tick * pos; ex = True
            if ex:
                pnl = q * (xp - ep) * pos - 0.0004 * q * (ep + xp); real += pnl
                T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = pnl; nt += 1; pos = 0; last = i
        if pos == 0 and posA[i] == 0 and sig[i] != 0 and i - last >= cd:
            pos = sig[i]; ep = c[i] + tick * pos; q = 500.0 / c[i]; ei = i; st = ep * (1.0 - pos * sl)
        if i % 30 == 0:
            add30[i // 30] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return add30, T[:nt]
FALSE = np.zeros(n, np.bool_)
def evaluate(lab, sig, exL=None, exS=None, H=240, sl=0.03, cd=0):
    add30, T = fill(o, h, l, c, posA, sig.astype(np.int8), FALSE if exL is None else exL, FALSE if exS is None else exS, H, sl, cd, TICK)
    if len(T) < 30: return None
    m_ = min(len(eq30), len(add30)); add30 = add30[:m_]; comb = eq30[:m_] + add30; dd = np.maximum.accumulate(comb) - comb; p = T[:, 3]; e = T[:, 0].astype(np.int64)
    isM = e < mid; gp = p[p > 0].sum(); gl = -p[p < 0].sum(); dly = np.diff(add30[::48])
    r = dict(v=lab, n=len(p), add=p.sum(), PF=gp / gl if gl > 0 else 99.0, win=(p > 0).mean(), addIS=p[isM].sum(), addOOS=p[~isM].sum(), nIS=int(isM.sum()),
             addCalm=p[isCalmBar[e]].sum(), add300=p[tms[e] >= t300].sum(), MaxDD=dd.max(), DD99=np.percentile(dd, 99), DD95=np.percentile(dd, 95),
             shA=dly.mean() / dly.std() * np.sqrt(365) if dly.std() > 0 else 0.0)
    for y in (2021, 2022, 2023, 2024, 2025, 2026): r[f'y{y}'] = p[pd.to_datetime(tms[e], unit='ms').year == y].sum() if False else 0.0
    yr = pd.to_datetime(tms[e], unit='ms').year.to_numpy()
    for y in (2021, 2022, 2023, 2024, 2025, 2026): r[f'y{y}'] = p[yr == y].sum()
    return r
def dirsig(longC, shortC, reg=None, ta=False):
    s = np.where(np.nan_to_num(longC).astype(bool), 1, np.where(np.nan_to_num(shortC).astype(bool), -1, 0)).astype(np.int8)
    if reg is not None: s = np.where(reg, s, 0)
    if ta: s = np.where(((s == 1) & up) | ((s == -1) & ~up), s, 0)
    return s.astype(np.int8)
# ── aileler: her biri (etiket, sinyal üretici) listesi ──
JOBS = []
HS = (60, 240, 720); SLS = (0.015, 0.03, 0.06); CDS = (0, 30)
for N, k, (rn, rg), ta, ex in itertools.product(ZN, (2.0, 2.5, 3.0, 3.5), REG.items(), (False, True), ('süre', 'ortalama')):
    JOBS.append((f'A zdönüş N{N} k{k} {rn} trend{int(ta)} çıkış:{ex}', (lambda N=N, k=k, rg=rg, ta=ta, ex=ex: (dirsig(ZN[N] <= -k, ZN[N] >= k, rg, ta), (ZN[N] >= 0) if ex == 'ortalama' else None, (ZN[N] <= 0) if ex == 'ortalama' else None))))
for N, md, (rn, rg), ta in itertools.product(DH, ('sön', 'takip'), REG.items(), (False, True)):
    JOBS.append((f'B donchian N{N} {md} {rn} trend{int(ta)}', (lambda N=N, md=md, rg=rg, ta=ta: ((dirsig(c < DL[N], c > DH[N], rg, ta) if md == 'sön' else dirsig(c > DH[N], c < DL[N], rg, ta)), None, None))))
for N, lv, (rn, rg), ta, ex in itertools.product(RS, (10, 15, 20), REG.items(), (False, True), ('süre', 'rsi50')):
    JOBS.append((f'C rsi{N} uç{lv} {rn} trend{int(ta)} çıkış:{ex}', (lambda N=N, lv=lv, rg=rg, ta=ta, ex=ex: (dirsig(RS[N] < lv, RS[N] > 100 - lv, rg, ta), (RS[N] >= 50) if ex == 'rsi50' else None, (RS[N] <= 50) if ex == 'rsi50' else None))))
for src, k, zt, md, lag_, (rn, rg) in itertools.product(('BTC', 'XMR'), (5, 15, 30, 60), (1.5, 2.0, 3.0), ('takip', 'sön'), (False, True), REG.items()):
    def g(src=src, k=k, zt=zt, md=md, lag_=lag_, rg=rg):
        zz = BZ[k] if src == 'BTC' else XZ[k]; rr = BR[k] if src == 'BTC' else XR[k]; zr = ZR[k]
        cond = (np.abs(zr) < 0.5 * np.abs(rr)) if lag_ else True
        lc, sc = (zz >= zt) & cond, (zz <= -zt) & cond
        return (dirsig(lc, sc, rg) if md == 'takip' else dirsig(sc, lc, rg)), None, None
    JOBS.append((f'D {src} öncü {k}dk z{zt} {md} geride{int(lag_)} {rn}', g))
for N, k, md, (rn, rg), ta in itertools.product(BS, (2.0, 2.5, 3.0), ('dönüş', 'takip'), REG.items(), (False, True)):
    JOBS.append((f'E basis z N{N} k{k} {md} {rn} trend{int(ta)}', (lambda N=N, k=k, md=md, rg=rg, ta=ta: ((dirsig(BS[N] <= -k, BS[N] >= k, rg, ta) if md == 'dönüş' else dirsig(BS[N] >= k, BS[N] <= -k, rg, ta)), None, None))))
for k, zt, ty, md, (rn, rg) in itertools.product(OZ, (2.0, 2.5, 3.0), ('boşalma', 'birikim'), ('dönüş', 'takip'), REG.items()):
    def g(k=k, zt=zt, ty=ty, md=md, rg=rg):
        oc = (OZ[k] <= -zt) if ty == 'boşalma' else (OZ[k] >= zt); pr = ZR[min(k, 240)]
        lc, sc = oc & (pr < 0), oc & (pr > 0)                         # fiyat düşmüş → dönüş long / takip short
        return (dirsig(lc, sc, rg) if md == 'dönüş' else dirsig(sc, lc, rg)), None, None
    JOBS.append((f'F OI {ty} {k}dk z{zt} {md} {rn}', g))
for vt, kk, md, (rn, rg), ta in itertools.product((3.0, 5.0, 8.0), (1, 5), ('sön', 'takip'), REG.items(), (False, True)):
    def g(vt=vt, kk=kk, md=md, rg=rg, ta=ta):
        dr = np.sign(c - L(c, kk)); sp = VZ >= vt
        return (dirsig(sp & (dr < 0), sp & (dr > 0), rg, ta) if md == 'sön' else dirsig(sp & (dr > 0), sp & (dr < 0), rg, ta)), None, None
    JOBS.append((f'G hacim sıçraması z{vt} {kk}bar {md} {rn} trend{int(ta)}', g))
for wn, rule, ft, (rn, rg) in itertools.product((15, 30, 60, 120), ('fundinge ters', 'funding yönü'), (0.0, 0.01, 0.03), REG.items()):
    def g(wn=wn, rule=rule, ft=ft, rg=rg):
        win = mins >= 480 - wn; sgn = np.sign(fund) * (np.abs(fund) >= ft)
        d_ = -sgn if rule == 'fundinge ters' else sgn
        return dirsig(win & (d_ > 0), win & (d_ < 0), rg), (mins < 5), (mins < 5)
    JOBS.append((f'H funding penceresi {wn}dk {rule} |f|≥{ft} {rn}', g))
fr60 = (L(c[::-1], 60)[::-1] / c - 1)
for tt, (rn, rg) in itertools.product((2.0, 2.5, 3.0), REG.items()):
    def g(tt=tt, rg=rg):
        dirH = np.zeros(24)
        for hh in range(24):
            m = (hr[:mid] == hh) & ((tms[:mid] // 60000) % 60 == 0); x = fr60[:mid][m]; x = x[~np.isnan(x)]
            if len(x) > 100 and abs(x.mean() / x.std() * np.sqrt(len(x))) >= tt: dirH[hh] = np.sign(x.mean())
        st_ = ((tms // 60000) % 60 == 0); d_ = dirH[hr] * st_
        return dirsig(d_ > 0, d_ < 0, rg), None, None
    JOBS.append((f'I saat deseni t≥{tt} (IS öğrenme) {rn}', g))
sq = S(c).rolling(240).std().to_numpy() / S(c).rolling(1440).std().to_numpy()
for N, sqt, (rn, rg) in itertools.product((60, 240, 720), (0.5, 0.7), REG.items()):
    JOBS.append((f'J sıkışma sonrası sön N{N} sq<{sqt} {rn}', (lambda N=N, sqt=sqt, rg=rg: (dirsig((c < DL[N]) & (sq < sqt), (c > DH[N]) & (sq < sqt), rg), None, None))))
    JOBS.append((f'J sıkışma kırılım takip N{N} sq<{sqt} {rn}', (lambda N=N, sqt=sqt, rg=rg: (dirsig((c > DH[N]) & (sq < sqt), (c < DL[N]) & (sq < sqt), rg), None, None))))
stt = Z['stt']
for N, k, ex in itertools.product(ZN, (1.5, 2.0, 2.5, 3.0), ('süre', 'ortalama')):
    JOBS.append((f'K trend içi geri çekilme N{N} k{k} çıkış:{ex}', (lambda N=N, k=k, ex=ex: (dirsig((ZN[N] <= -k) & (stt > 0), (ZN[N] >= k) & ~up & (stt <= 0)), (ZN[N] >= 0) if ex == 'ortalama' else None, (ZN[N] <= 0) if ex == 'ortalama' else None))))
nv = len(JOBS) * len(HS) * len(SLS) * len(CDS)
print(f'[{W}] aile işleri {len(JOBS)} · toplam varyant {nv}', flush=True)
rows = []; t1 = time.time()
for j, (lab, gen) in enumerate(JOBS):
    if j % NW != W: continue
    s, eL, eS = gen()
    if not (s != 0).any(): continue
    for H, sl, cd in itertools.product(HS, SLS, CDS):
        r = evaluate(f'{lab} | tut{H} stop%{sl * 100:g} bekle{cd}', s, None if eL is None else np.nan_to_num(eL).astype(np.bool_), None if eS is None else np.nan_to_num(eS).astype(np.bool_), H, sl, cd)
        if r: rows.append(r)
    if j % 40 == W: print(f'[{W}] {j}/{len(JOBS)} {time.time() - t1:.0f} sn · sonuç {len(rows)}', flush=True)
pd.DataFrame(rows).to_pickle(f'/home/claude/zecwork/hunt_{W}.pkl'); print(f'[{W}] bitti {len(rows)} sonuç {time.time() - t0:.0f} sn', flush=True)
