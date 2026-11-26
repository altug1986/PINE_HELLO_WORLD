exec(open('/home/claude/zecwork/maske7.py').read().split("S3, A3 = mk('3h')")[0])
@njit(cache=False)
def comb3(o, h, l, c, tick, rp, rx, S, A, athr, capP):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); fill = False; cap = 0.0; nE = 0; nH = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if fill and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick)
            elif (not fill) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0: rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0; fill = False
        v = rp[i]; s = int(S[i]); tg = v
        if tg == 0 and s != 0 and A[i] >= athr: tg = s
        nf = tg != 0 and v == 0
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if nf: cap = ep * (1.0 - capP * tg); nE += 1
        elif nf and not fill: cap = c[i] * (1.0 - capP * tg); nH += 1   # V3.3 → ek sistem devri: stop devir fiyatından
        fill = nf
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nE, nH
S3, A3 = mk('3h')
for z in (36.0, 38.0, 40.0, 42.0):
    for c_ in (0.04, 0.05, 0.06):
        e, nE, nH = comb3(o_, h_, l_, cc, 0.01, rp, rx, S3, A3, z, c_); pr(f'ADX≥{z:.0f} stop %{c_ * 100:.0f} (giriş {nE}, devir {nH})', met(e, ''))
e, _, _ = comb3(o_, h_, l_, cc, 0.01, rp, rx, S3, A3, 40.0, 0.05); m = met(e, ''); print({k: round(float(v), 2) for k, v in m.items() if k != 'v'}); print({k: round(float(B[k]), 2) for k in m if k != 'v'})
x = pd.Series(e - e0, ixh).resample('YE').last(); print('ek katkı yıllık:', dict(zip(x.index.year, np.diff(np.r_[0, x.values]).round(0))))
t23 = np.searchsorted(tms, 1675382400000); print(f'TV penceresi: V3.3 {e0[-1] - e0[t23]:.1f} → V3.4 {e[-1] - e[t23]:.1f}')
