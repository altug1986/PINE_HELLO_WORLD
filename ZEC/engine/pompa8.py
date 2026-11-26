# POMPA çıkış / yeniden giriş + 20 dk ADX: erken çıkış sadece 20 dk ADX ≥ eşik & −DI > +DI iken; yeniden long sadece 20 dk ADX ≥ eşik & +DI > −DI iken
exec(open('/home/claude/zecwork/pompa7.py').read().split("t0 = time.time(); out = []")[0].replace("ev, evE, evX, W, H, g, mode, sR):", "ev, evE, evX, W, H, g, mode, sR, okR):").replace("c[i] >= trg:", "c[i] >= trg and okR[i]:"))
def dmi20(m=20):
    b = day * 1440 + (mod // m) * m; g = pd.DataFrame({'b': b, 'h': h_, 'l': l_, 'c': cc}).groupby('b', sort=True).agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    ct = g.index.values + m; H, L, C = g.h.values, g.l.values, g.c.values; pc = np.r_[C[0], C[:-1]]; tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc)))
    up = np.r_[0, np.diff(H)]; dn = np.r_[0, -np.diff(L)]; at = rma(tr, 14); p = 100 * rma(np.where((up > dn) & (up > 0), up, 0), 14) / at; mm = 100 * rma(np.where((dn > up) & (dn > 0), dn, 0), 14) / at
    ax = rma(100 * abs(p - mm) / np.where(p + mm == 0, 1, p + mm), 14); j = np.searchsorted(ct, mn, side='right') - 1; ok = j >= 0; j = np.maximum(j, 0)
    return np.where(ok, ax[j], 0.0), np.where(ok, np.sign(p - mm)[j], 0.0)
eA, eD = dmi20()
src3 = open('/home/claude/zecwork/cikis.py').read(); a0 = src3.index('@njit(cache=False)\ndef simX'); b0 = src3.index('    return eq, T[:nt]', a0) + len('    return eq, T[:nt]')
sw = src3[a0:b0].replace('def simX(', 'def simW(').replace('byp, cap, oppN', 'byp, capL, capS, eA, eD, eThr, eDrop, oppN').replace('cap * ep', '(capL if pos == 1 else capS) * ep')
anc = "T[nt, 6] = 1; T[nt, 7] = own; nt += 1; pos = 0; own = -1; oc = 0\n"; k0 = sw.index(anc) + len(anc)
sw = sw[:k0] + """        if pos == 1 and own == 5 and eThr > 0 and c[i] <= ep * (1 - eDrop) and eA[i] >= eThr and eD[i] < 0:
            xp = c[i] - 0.01; pnl = q * (xp - ep) - 0.0004 * q * (ep + xp); real += pnl
            T[nt, 0] = ei; T[nt, 1] = i; T[nt, 2] = pos; T[nt, 3] = ep; T[nt, 4] = xp; T[nt, 5] = pnl; T[nt, 6] = 5; T[nt, 7] = own; nt += 1; pos = 0; own = -1; oc = 0
""" + sw[k0:]
exec(sw)
def chainW(eThr, eDrop):
    eqb, Tb = simW(*args, *XV, TS6, THv, spk, 60, 3, byp, 0.10, 0.04, eA, eD, eThr, eDrop, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    rp_, rx_, rn_ = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08); return rp_, rx_, Tb
out = []; t0 = time.time()
for eThr, eDrop in [(0.0, 0.0)] + [(z, dp) for z in (20.0, 25.0, 30.0) for dp in (0.01, 0.02, 0.03)]:
    rp_, rx_, Tb = chainW(eThr, eDrop); pm = Tb[(Tb[:, 7] == 5) & (Tb[:, 2] == 1)]; st = pm[np.isin(pm[:, 6], (1, 5))]
    ev = np.zeros(n, np.bool_); evE = np.zeros(n); evX = np.zeros(n)
    for r in st: j = int(r[1]); ev[j] = True; evE[j] = r[3]; evX[j] = r[4]
    for rThr in (0.0, 20.0, 25.0, 30.0):
        okR = ((eA >= rThr) & (eD > 0)) if rThr > 0 else np.ones(n, np.bool_)
        for md, g, H in [(0, 0.0, 0)] + [(2, g, H) for g in (0.01, 0.02, 0.03) for H in (240, 480)]:
            if md == 0 and rThr > 0: continue
            e, nR, pR = combR(o_, h_, l_, cc, 0.01, rp_, rx_, S9, A9, 38.0, 0.05, ev, evE, evX, 720, H, g, md, 0.05, okR); m = met(e, '')
            out.append((eThr, eDrop, rThr, md, g, H, m['net'] - 11023.4, m['IS'] - 5577.7, m['OOS'] - 5445.7, m['MaxDD'], m['DD99'], m['Calmar'], len(pm), pm[:, 5].sum(), (pm[:, 6] == 5).sum(), nR, pR))
    print(eThr, eDrop, f'{time.time() - t0:.0f}s', flush=True)
d = pd.DataFrame(out, columns=['cADX', 'cDüşüş', 'gADX', 'tetik', 'g', 'tutma', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'pompaN', 'pompaNet', 'erkenÇıkış', 'yeniden', 'yenidenNet']); d.to_csv('pompa8.csv', index=False)
