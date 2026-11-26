# Kanama-2: giriş ERTELE (öbek süpürülünce V3.0 hâlâ aynı yöndeyse daha iyi fiyattan katıl) + yön ayrımı; seçim: ilk yarı kanaması, şart: ilk yarı GENEL net düşmesin
exec(open('/home/claude/zecwork/kanama.py').read().split("rows = []; T0b = time.time()")[0])
@njit(cache=False)
def ovD(o, h, l, c, tick, rp, rx, rn, dS, dL, nH, nL):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); lv = 0.0; pend = 0; nd = 0
    for i in range(1, n):
        if real != 0 and real == rp[i - 1] and rx[i] > 0: rz += q * (rx[i] - ep) * real - 0.0004 * q * (ep + rx[i]); real = 0
        v = rp[i]; tg = real
        if rn[i]:
            pend = 0; tg = v
            if (v < 0 and dS[i] and not np.isnan(nH[i])) or (v > 0 and dL[i] and not np.isnan(nL[i])): pend = v; lv = nH[i] if v < 0 else nL[i]; tg = 0; nd += 1
        elif v == 0: tg = 0; pend = 0
        elif pend != 0:
            if v != pend: pend = 0; tg = 0
            elif (pend < 0 and h[i] >= lv) or (pend > 0 and l[i] <= lv): pend = 0; tg = v
            else: tg = 0
        elif real != v: tg = 0
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nd
rows = []
for Lb in (30, 60, 120, 240):
    nH, nL, kH, kL, pa = liq(h, l, c, Lb, Lb, 0.01); dH = (nH / c - 1) * 100; dL = (1 - nL / c) * 100
    for X, kmin, usePA, side in itertools.product((0.5, 1.0, 2.0, 3.0), (1, 2), (True, False), ('ikisi', 'sadece long', 'sadece short')):
        sS = np.nan_to_num((dH <= X) & (kH >= kmin), nan=False) & ((pa > 0) if usePA else True); sL = np.nan_to_num((dL <= X) & (kL >= kmin), nan=False) & ((pa < 0) if usePA else True)
        if side == 'sadece long': sS = F0
        if side == 'sadece short': sL = F0
        for kural in ('veto', 'ertele (süpürülünce katıl)'):
            e, nc = ovPA(o, h, l, c, TICK, rp, rx, rn, sS, sL, F0, F0, nH, nL) if kural == 'veto' else ovD(o, h, l, c, TICK, rp, rx, rn, sS, sL, nH, nL)
            bl = bleed(e); r = ev(e, nc, 0, 0, pivot=Lb, mesafe=X, küme=kmin, PAşart=usePA, yön=side, kural=kural); r['kanamaİS'] = bl[isE].sum(); r['kanamaOOS'] = bl[~isE].sum(); rows.append(r)
R = pd.DataFrame(rows); B = ev(e30, 0, 0, 0)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R.to_pickle('/home/claude/zecwork/kanama2.pkl'); pd.set_option('display.width', 300)
c_ = ['pivot', 'mesafe', 'küme', 'PAşart', 'yön', 'kural', 'işlem', 'kanamaİS', 'kanamaOOS', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']
ok = R[R.dIS >= 0]; print(f'{len(R)} varyant | ilk yarı genel neti düşürmeyen {len(ok)} | kapı (≥{B["net"] * 1.06:.0f} & MaxDD≤{B["MaxDD"]:.1f} & Ulcer≤{B["Ulcer"]:.2f} & iki yarı): {((R.net >= B["net"] * 1.06) & (R.MaxDD <= B["MaxDD"] + 0.01) & (R.Ulcer <= B["Ulcer"]) & (R.dIS > 0) & (R.dOOS > 0)).sum()}')
print('2) İLK YARI KANAMASINA GÖRE SEÇİM (şart: ilk yarı genel net düşmez) → ikinci yarı ve GENEL:'); print(ok.sort_values('kanamaİS', ascending=False)[c_].head(8).round(1).to_string(index=False))
print('GENEL NETE GÖRE EN İYİ 6:'); print(R.sort_values('dnet', ascending=False)[c_].head(6).round(1).to_string(index=False))
print(R.groupby(['kural', 'yön']).agg(medKanİS=('kanamaİS', 'median'), medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('dOOS', lambda s: ((s > 0) & (R.loc[s.index, 'dIS'] > 0)).sum())).round(0).to_string())
