# ULCER BÖLGELERİ: V2.9 sualtı haritası + TERS BALİNA (balina baskısı / top trader yığılması) + MİKRO FORMASYON (ikili tepe-dip, OBO/TOBO) dönüş modülleri
import sys, glob, time, itertools, numpy as np, pandas as pd, numba
sys.path.insert(0, '/home/claude/zecwork'); sys.argv = ['x', 'yok']
import pfsearch as P
from pfsearch import o, h, l, c, posA, bEp, bXp, bXd, R10, tms, n, TICK, ix, S
mid = int(np.load('/home/claude/zecwork/base28.npz')['mid'])
cols = ['time', 'wh_buy_q50k', 'wh_sell_q50k', 'wh_buy_q200k', 'wh_sell_q200k', 'tt_pos_ls', 'tt_acc_ls']
M = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))])
M.index = pd.to_datetime(M.time, unit='ms') if M.time.dtype.kind in 'iu' else pd.to_datetime(M.time); M = M[~M.index.duplicated()].reindex(ix)
WF = (M.wh_buy_q50k.fillna(0) - M.wh_sell_q50k.fillna(0)).to_numpy(); print('balina kapsama %', round(M.wh_buy_q50k.notna().mean() * 100, 1), flush=True)
TT = M.tt_pos_ls.ffill().shift(5).to_numpy()
# ── V2.9 sualtı haritası ──
e9 = P.ov5(o, h, l, c, posA, bEp, bXp, bXd, R10, P.F0, 0, 0, 0.0, P.ZR60 < 0, P.F0, 0, TICK)
s30 = S(e9, ix)[::30]; dd30 = (s30.cummax() - s30)
ZONE = np.zeros(n, np.bool_); zz = (S(e9, ix).cummax() - S(e9, ix)).to_numpy() >= 100; ZONE[:] = zz      # 100 $+ sualtı (değerlendirme için)
def ulcer(e):
    x = e[::30]; d = (np.maximum.accumulate(x) - x) / 5.0; return np.sqrt((d ** 2).mean())
print(f'V2.9: Ulcer {ulcer(e9):.2f} · 100$+ sualtı süre %{ZONE.mean() * 100:.1f}', flush=True)
@numba.njit(cache=True)
def ov7(o, h, l, c, posA, bEp, bXp, bXd, R10, sig, sstop, mode, H, slp, tick):
    """V2.9 (gölge + RSI10 ters) + SİNYAL MOTORU: sig[i]=±1 (bar kapanışı). mode 1: yalnız ters (baz ters yöndeyken), 2: yalnız flat iken, 3: ikisi.
    Sinyal pozisyonu H bar sürer; stop = sstop[i] (formasyon ucu) varsa o, yoksa giriş ± slp."""
    n = len(c); pos = 0; ep = 0.0; q = 0.0; real = 0.0; mode2 = 0; prevD = 0; tPx = 0.0; tBar = 0; tSide = 0; rStp = 0.0
    sOwn = False; sBar = 0; sStp = 0.0; sDir = 0; eq = np.zeros(n); ns = 0
    for i in range(1, n):
        if pos != 0:
            xp = -1.0
            if sOwn:
                if (pos == 1 and l[i] <= sStp) or (pos == -1 and h[i] >= sStp):
                    xp = (min(o[i], sStp) - tick) if pos == 1 else (max(o[i], sStp) + tick); sOwn = False
            elif mode2 == 2 and pos == -tSide:
                if (pos == 1 and l[i] <= rStp) or (pos == -1 and h[i] >= rStp):
                    xp = (min(o[i], rStp) - tick) if pos == 1 else (max(o[i], rStp) + tick); mode2 = 0
            elif not np.isnan(bXp[i]) and pos == bXd[i]:
                xp = bXp[i]
            if xp > 0:
                real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
        D = posA[i]; want = D
        if D != prevD: mode2 = 0
        elif D != 0:
            if mode2 == 0:
                if (D > 0 and R10[i] >= 94) or (D < 0 and R10[i] <= 6):
                    want = -D; mode2 = 2; tPx = c[i]; tBar = i; tSide = D
            else:
                bi = (R10[i] <= 70 or i - tBar >= 240 or c[i] >= tPx * 1.03) if tSide > 0 else (R10[i] >= 30 or i - tBar >= 240 or c[i] <= tPx * 0.97)
                if bi: mode2 = 0
                else: want = -tSide if pos == -tSide else 0
        prevD = D
        if sOwn and i - sBar >= H: sOwn = False
        if sOwn: want = sDir
        elif sig[i] != 0:
            ok = (mode != 2 and pos == -sig[i]) or (mode != 1 and pos == 0) or (mode == 3 and D == -sig[i])
            if ok:
                want = sig[i]; sOwn = True; sBar = i; sDir = sig[i]; ns += 1
                sStp = sstop[i] if not np.isnan(sstop[i]) else c[i] * (1.0 - sig[i] * slp)
        if want != pos:
            if pos != 0:
                xp = c[i] - tick * pos; real += q * (xp - ep) * pos - 0.0004 * q * (ep + xp); pos = 0
            if want != 0:
                pos = want; ep = c[i] + tick * pos; q = 500.0 / c[i]
                if not sOwn and mode2 == 2 and want == -tSide: rStp = ep * (1.0 - want * 0.02)
        eq[i] = real + (q * (c[i] - ep) * pos - 0.0004 * q * ep if pos != 0 else 0.0)
    return eq, ns
# ── mikro formasyon (5 / 15 dk) ──
@numba.njit(cache=True)
def formations(H_, L_, C_, A_, k, tol, maxSpan, kind):
    m = len(C_); sig = np.zeros(m, np.int8); stp = np.full(m, np.nan)
    ph = np.full(m, np.nan); pl = np.full(m, np.nan)                       # onaylı pivot (j+k anında bilinir)
    hiI = np.zeros(4, np.int64); hiV = np.zeros(4); nh = 0; loI = np.zeros(4, np.int64); loV = np.zeros(4); nl = 0
    for t in range(2 * k, m):
        j = t - k; isH = True; isL = True
        for u in range(j - k, j + k + 1):
            if H_[u] > H_[j]: isH = False
            if L_[u] < L_[j]: isL = False
        if isH:
            for z in range(3): hiI[z] = hiI[z + 1]; hiV[z] = hiV[z + 1]
            hiI[3] = j; hiV[3] = H_[j]; nh += 1
        if isL:
            for z in range(3): loI[z] = loI[z + 1]; loV[z] = loV[z + 1]
            loI[3] = j; loV[3] = L_[j]; nl += 1
        a = A_[t]
        if np.isnan(a) or a <= 0: continue
        # ikili tepe → short / ikili dip → long (boyun kırılımı)
        if (kind == 0 or kind == 2) and nh >= 2:
            p1, p2 = hiI[2], hiI[3]
            if t - p1 <= maxSpan and p2 - p1 >= 2 * k and abs(hiV[3] - hiV[2]) <= tol * a:
                neck = 1e18
                for u in range(p1, p2 + 1): neck = min(neck, L_[u])
                if C_[t] < neck and C_[t - 1] >= neck: sig[t] = -1; stp[t] = max(hiV[2], hiV[3]) + 0.25 * a
        if (kind == 0 or kind == 2) and nl >= 2 and sig[t] == 0:
            p1, p2 = loI[2], loI[3]
            if t - p1 <= maxSpan and p2 - p1 >= 2 * k and abs(loV[3] - loV[2]) <= tol * a:
                neck = -1e18
                for u in range(p1, p2 + 1): neck = max(neck, H_[u])
                if C_[t] > neck and C_[t - 1] <= neck: sig[t] = 1; stp[t] = min(loV[2], loV[3]) - 0.25 * a
        # OBO (omuz-baş-omuz) / TOBO
        if (kind == 1 or kind == 2) and nh >= 3 and sig[t] == 0:
            l_, hd, r_ = hiV[1], hiV[2], hiV[3]
            if t - hiI[1] <= maxSpan and hd > max(l_, r_) + 0.5 * tol * a and abs(l_ - r_) <= tol * a:
                neck = 1e18
                for u in range(hiI[1], hiI[3] + 1): neck = min(neck, L_[u])
                if C_[t] < neck and C_[t - 1] >= neck: sig[t] = -1; stp[t] = r_ + 0.25 * a
        if (kind == 1 or kind == 2) and nl >= 3 and sig[t] == 0:
            l_, hd, r_ = loV[1], loV[2], loV[3]
            if t - loI[1] <= maxSpan and hd < min(l_, r_) - 0.5 * tol * a and abs(l_ - r_) <= tol * a:
                neck = -1e18
                for u in range(loI[1], loI[3] + 1): neck = max(neck, H_[u])
                if C_[t] > neck and C_[t - 1] <= neck: sig[t] = 1; stp[t] = r_ - 0.25 * a
    return sig, stp
def to1m(tf, sig_tf, stp_tf, last_idx):
    s1 = np.zeros(n, np.int8); st1 = np.full(n, np.nan); s1[last_idx] = sig_tf; st1[last_idx] = stp_tf; return s1, st1
TF = {}
for tf in (5, 15):
    g = S(np.arange(n), ix).groupby(ix.floor(f'{tf}min')); first = g.min().to_numpy(); last = g.max().to_numpy()
    Hh = np.maximum.reduceat(h, first); Ll = np.minimum.reduceat(l, first); Cc = c[last]
    tr = np.maximum(Hh - Ll, np.maximum(np.abs(Hh - np.r_[Cc[0], Cc[:-1]]), np.abs(Ll - np.r_[Cc[0], Cc[:-1]]))); Aa = S(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    TF[tf] = (Hh, Ll, Cc, Aa, last)
rows = []; t0 = time.time()
def ev(lab, sig, stp, mode, H, slp):
    e, ns = ov7(o, h, l, c, posA, bEp, bXp, bXd, R10, sig.astype(np.int8), stp, mode, H, slp, TICK)
    x = e[::30]; dd = np.maximum.accumulate(x) - x; d = e - e9; dl = np.diff(np.r_[0, d])
    rows.append(dict(v=lab, n=ns, net=e[-1], dNet=e[-1] - e9[-1], dIS=d[mid], dOOS=(e[-1] - e[mid]) - (e9[-1] - e9[mid]), dZone=dl[ZONE].sum(), Ulcer=ulcer(e), MaxDD=dd.max(), DD99=np.percentile(dd, 99)))
NAN = np.full(n, np.nan)
ev('V2.9', np.zeros(n, np.int8), NAN, 3, 60, 0.02)
for tf, k, tol, kind, span in itertools.product((5, 15), (2, 3), (0.5, 1.0), (0, 1, 2), (48, 96)):
    Hh, Ll, Cc, Aa, last = TF[tf]; sg, sp = formations(Hh, Ll, Cc, Aa, k, tol, span, kind); s1, st1 = to1m(tf, sg, sp, last)
    for mode, H, stopk in itertools.product((1, 2, 3), (60, 240), ('formasyon', '%2')):
        ev(f'FORM {tf}dk k{k} tol{tol} {["ikili", "OBO", "ikisi"][kind]} açıklık{span} | mod{mode} tut{H} stop:{stopk}', s1, st1 if stopk == 'formasyon' else NAN, mode, H, 0.02)
print(f'formasyon bitti {time.time() - t0:.0f} sn', flush=True)
for N, kz, var in itertools.product((15, 60, 240), (2.0, 3.0), ('emilim', 'karşıt', 'takip')):
    f_ = S(WF).rolling(N).sum(); z = (f_ / f_.rolling(43200, min_periods=10000).std()).to_numpy(); pr = np.r_[np.full(N, np.nan), c[N:] / c[:-N] - 1]
    if var == 'emilim': sg = np.where((z >= kz) & (pr <= 0), -1, np.where((z <= -kz) & (pr >= 0), 1, 0))
    elif var == 'karşıt': sg = np.where(z >= kz, -1, np.where(z <= -kz, 1, 0))
    else: sg = np.where(z >= kz, 1, np.where(z <= -kz, -1, 0))
    for mode, H, slp in itertools.product((1, 2, 3), (60, 240), (0.02, 0.04)):
        ev(f'BALİNA 50k akış {N}dk z{kz} {var} | mod{mode} tut{H} sl%{slp * 100:g}', np.nan_to_num(sg), NAN, mode, H, slp)
for W_, kz, var in itertools.product((1440, 4320), (1.5, 2.0, 2.5), ('karşıt', 'takip')):
    z = ((S(TT) - S(TT).rolling(W_, min_periods=W_ // 2).mean()) / S(TT).rolling(W_, min_periods=W_ // 2).std()).to_numpy()
    sg = np.where(z >= kz, -1, np.where(z <= -kz, 1, 0)) if var == 'karşıt' else np.where(z >= kz, 1, np.where(z <= -kz, -1, 0))
    sg = np.where(np.r_[0, sg[:-1]] == sg, 0, sg)                         # yalnız ilk bar
    for mode, H, slp in itertools.product((1, 2, 3), (60, 240), (0.02, 0.04)):
        ev(f'TOP TRADER oran z({W_}) {kz} {var} | mod{mode} tut{H} sl%{slp * 100:g}', np.nan_to_num(sg), NAN, mode, H, slp)
R = pd.DataFrame(rows); R.to_pickle('/home/claude/zecwork/ulcer1.pkl'); print(f'bitti {len(R)} varyant {time.time() - t0:.0f} sn')
