# POMPA: daha erken çık (sıkı stop) + piyasa döndüyse tekrar long (yeniden giriş), V3.5 (190 dk ek sistem) üstüne
exec(open('/home/claude/zecwork/pompa5.py').read().split("R = {}")[0])
import itertools, time
def chainP(ts5):
    TSv = TS6.copy(); TSv[5] = ts5
    eqb, Tb = simZ(*args, *XV, TSv, THv, spk, 60, 3, byp, 0.10, 0.04, 0.0, 1, False, False, False)
    P_ = np.zeros(n, np.int8); bXp = np.full(n, np.nan); bXd = np.zeros(n, np.int8)
    for r in Tb: P_[int(r[0]):int(r[1])] = int(r[2]); bXp[int(r[1])] = r[4]; bXd[int(r[1])] = int(r[2])
    vp = np.zeros(n, np.int8); vx = np.zeros(n); vn = np.zeros(n, np.bool_); TR = np.zeros((20000, 4)); run29(args[0], args[1], args[2], cc, P_, bXp, bXd, R10, 0.01, vp, vx, vn, TR)
    rp_, rx_, rn_ = rec30(args[0], args[1], args[2], cc, 0.01, vp, vx, vn, EVT, 1440, 0.08); return rp_, rx_, Tb
@njit(cache=False)
def combR(o, h, l, c, tick, rp, rx, S, A, athr, capP, ev, evE, evX, W, H, g, mode, sR):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); fill = 0; cap = 0.0; rEnd = -1; wEnd = -1; trg = 0.0; nR = 0; pR = 0.0; rEp = 0.0; rq = 0.0
    for i in range(1, n):
        if ev[i] and mode > 0: wEnd = i + W; trg = evE[i] if mode == 1 else evX[i] * (1 + g); rEnd = -1   # POMPA stop çıkışı → izleme penceresi
        if real != 0:
            xp = -1.0
            if fill != 0 and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)):
                xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick)
                if fill == 2: rEnd = -1
            elif fill == 0 and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn; real = 0
                if fill == 2: pR += pn
                fill = 0
        v = rp[i]; tg = v; nf = 0
        if v == 0 and wEnd >= i and rEnd < 0 and c[i] >= trg: rEnd = i + H; wEnd = -1; nR += 1          # piyasa döndü → tekrar long
        if v == 0 and rEnd >= i: tg = 1; nf = 2
        elif v == 0 and S[i] != 0 and A[i] >= athr: tg = int(S[i]); nf = 1
        if v != 0: rEnd = -1
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; pn = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += pn
                if fill == 2: pR += pn
                real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if nf != 0: cap = ep * (1.0 - (capP if nf == 1 else sR) * tg)
        elif nf != 0 and nf != fill: cap = c[i] * (1.0 - (capP if nf == 1 else sR) * tg)
        fill = nf
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nR, pR
t0 = time.time(); out = []
for ts5 in (3.0, 4.0, 5.0, 6.0, 8.0):
    rp_, rx_, Tb = chainP(ts5); pm = Tb[(Tb[:, 7] == 5) & (Tb[:, 2] == 1)]; st = pm[pm[:, 6] == 1]
    ev = np.zeros(n, np.bool_); evE = np.zeros(n); evX = np.zeros(n)
    for r in st: j = int(r[1]); ev[j] = True; evE[j] = r[3]; evX[j] = r[4]
    grid = [(0, 0, 0, 0.0, 0.0)] + [(md, W, H, g, s) for md, g in ((1, 0.0), (2, 0.005), (2, 0.01), (2, 0.02), (2, 0.03)) for W, H, s in itertools.product((60, 240, 720), (240, 480), (0.03, 0.05))]
    for md, W, H, g, s in grid:
        e, nR, pR = combR(o_, h_, l_, cc, 0.01, rp_, rx_, S9, A9, 38.0, 0.05, ev, evE, evX, W, H, g, md, s); m = met(e, '')
        out.append((ts5, md, W, H, g, s, m['net'] - 11023.4, m['IS'] - 5577.7, m['OOS'] - 5445.7, m['MaxDD'], m['DD99'], m['Calmar'], len(pm), pm[:, 5].sum(), len(st), nR, pR))
    print(ts5, f'{time.time() - t0:.0f}s', flush=True)
d = pd.DataFrame(out, columns=['ATRx', 'tetik', 'pencere', 'tutma', 'g', 'stopR', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'pompaN', 'pompaNet', 'pompaStop', 'yeniden', 'yenidenNet']); d.to_csv('pompa7.csv', index=False)
