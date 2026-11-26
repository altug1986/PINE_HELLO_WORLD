# ÜST ZAMAN DİLİMİ MASKE SİSTEMİ: önce tek başına çalışan 1 s / 4 s / günlük trend sistemleri, sonra V3.3 ile birleştir:
# M1 ters girişi engelle · M2 ters pozisyonu kes · M3 V3.3 FLAT iken üst sistemi takip et · M4 = M1+M3 · M5 = M3 sadece güçlü trendde (ADX≥25)
exec(open('/home/claude/zecwork/winrate.py').read().split("dur = TT[:, 1] - TT[:, 0]")[0])
o_, h_, l_ = args[0], args[1], args[2]
def bars(rule): return pd.DataFrame({'o': pd.Series(o_, ixh).resample(rule).first(), 'h': pd.Series(h_, ixh).resample(rule).max(), 'l': pd.Series(l_, ixh).resample(rule).min(), 'c': pd.Series(cc, ixh).resample(rule).last()}).dropna()
def m1(v, D, td): s_ = pd.Series(np.asarray(v, float), D.index + td); j = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; out = s_.values[np.maximum(j, 0)]; out[j < 0] = 0; return np.nan_to_num(out)
def rma(x, N): return pd.Series(x).ewm(alpha=1 / N, adjust=False).mean().values
def adx(D):
    H, L, C = D.h.values, D.l.values, D.c.values; pc = np.r_[C[0], C[:-1]]; tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc))); up = np.r_[0, np.diff(H)]; dn = np.r_[0, -np.diff(L)]; at = rma(tr, 14)
    p = 100 * rma(np.where((up > dn) & (up > 0), up, 0), 14) / at; m = 100 * rma(np.where((dn > up) & (dn > 0), dn, 0), 14) / at; return rma(100 * abs(p - m) / np.where(p + m == 0, 1, p + m), 14)
SYS = {}
for rule, td, lab in (('1h', pd.Timedelta('1h'), '1 s'), ('4h', pd.Timedelta('4h'), '4 s'), ('1D', pd.Timedelta('1D'), 'günlük')):
    D = bars(rule); C = D.c; ax = m1(adx(D), D, td)
    for f, s in ((10, 30), (20, 50), (50, 200)):
        st = np.sign(C.ewm(span=f, adjust=False).mean() - C.ewm(span=s, adjust=False).mean()).values; SYS[f'{lab} EMA{f}/{s}'] = (m1(st, D, td), ax)
    if rule != '1h':
        for ne, nx in ((20, 10), (55, 20)):
            hi = D.h.rolling(ne).max().shift(1); lo = D.l.rolling(ne).min().shift(1); hx = D.h.rolling(nx).max().shift(1); lx = D.l.rolling(nx).min().shift(1); st = np.zeros(len(D)); p = 0
            for t in range(len(D)):
                if p == 1 and C.iloc[t] < lx.iloc[t]: p = 0
                elif p == -1 and C.iloc[t] > hx.iloc[t]: p = 0
                if p == 0 and C.iloc[t] > hi.iloc[t]: p = 1
                elif p == 0 and C.iloc[t] < lo.iloc[t]: p = -1
                st[t] = p
            SYS[f'{lab} Donchian {ne}/{nx}'] = (m1(st, D, td), ax)
    if rule != '1D':
        H, L = D.h.values, D.l.values; pc = np.r_[C.values[0], C.values[:-1]]; at = rma(np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc))), 10); hl = (H + L) / 2; up = hl - 3 * at; dn = hl + 3 * at
        st = np.zeros(len(D)); fu, fd, d_ = up[0], dn[0], 1
        for t in range(1, len(D)):
            fu = max(up[t], fu) if C.values[t - 1] > fu else up[t]; fd = min(dn[t], fd) if C.values[t - 1] < fd else dn[t]
            d_ = 1 if C.values[t] > fd else (-1 if C.values[t] < fu else d_); st[t] = d_
        SYS[f'{lab} SuperTrend 10/3'] = (m1(st, D, td), ax)
@njit(cache=False)
def solo(c, pos, tick):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n)
    for i in range(1, n):
        tg = int(pos[i])
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq
@njit(cache=False)
def comb(o, h, l, c, tick, rp, rx, rn, S, A, mode, athr, capP):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); fill = False; cap = 0.0; skip = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if fill and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick); fill = False
            elif (not fill) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0: rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
        v = rp[i]; s = int(S[i])
        if rn[i]: skip = 0
        tg = v
        if mode in (1, 4) and rn[i] and v != 0 and s == -v: skip = v
        if mode == 2 and v != 0 and s == -v: tg = 0
        if skip != 0 and v == skip: tg = 0
        if mode in (3, 4, 5) and tg == 0 and v == 0 and s != 0 and (mode != 5 or A[i] >= athr): tg = s
        fill = tg != 0 and v == 0
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if fill: cap = ep * (1.0 - capP * tg)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq
F0 = np.zeros(n); e0 = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, F0, F0, 0, 0.0, 0.1); B = met(e0, 'V3.3'); print(f"V3.3 {e0[-1]:.1f}")
rows = []
for nm, (S, A) in SYS.items():
    es = solo(cc, S, 0.01); x = es[::30]; dd = (np.maximum.accumulate(x) - x).max(); r0 = dict(sistem=nm, tekNet=es[-1], tekIS=es[midI], tekOOS=es[-1] - es[midI], tekMaxDD=dd)
    for md, lab in ((1, 'M1 ters girişi engelle'), (2, 'M2 ters pozisyonu kes'), (3, 'M3 FLAT iken üst sistemi takip'), (4, 'M1+M3'), (5, 'M3 sadece ADX≥25')):
        e = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S, A, md, 25.0, 0.10); rows.append({**r0, **met(e, lab)})
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/maske.pkl'); pd.set_option('display.width', 300)
print(R.drop_duplicates('sistem')[['sistem', 'tekNet', 'tekIS', 'tekOOS', 'tekMaxDD']].round(0).to_string(index=False))
print(f"\nbirleşim {len(R)} | iki yarı artı {R.ikiYarı.sum()} | kapı (≥{B['net'] * 1.06:.0f}, MaxDD≤{B['MaxDD']:.1f}, iki yarı): {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & R.ikiYarı).sum()}")
print(R.groupby('v').agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum')).round(0).to_string())
print(R.sort_values('dnet', ascending=False)[['sistem', 'v', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].head(10).round(1).to_string(index=False))
