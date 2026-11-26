exec(open('/home/claude/zecwork/maske7.py').read().split("S3, A3 = mk('3h')")[0])
@njit(cache=False)
def comb2(o, h, l, c, tick, rp, rx, rn, S, A, athr, capP, blkOn):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); fill = False; cap = 0.0; blk = 0; nE = 0; nS = 0; nT = 0; wins = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0; isF = fill
            if fill and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)):
                xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick); fill = False; nS += 1
                if blkOn: blk = real
            elif (not fill) and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0:
                p = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += p; real = 0
                if isF: nT += 1; wins += p > 0
        v = rp[i]; s = int(S[i])
        if blk != 0 and (s != blk or A[i] < athr or v != 0): blk = 0
        tg = v
        if tg == 0 and s != 0 and A[i] >= athr and s != blk: tg = s
        nf = tg != 0 and v == 0
        if tg != real:
            if real != 0:
                xp = c[i] - tick * real; p = q * (xp - ep) * real - 0.0004 * q * (ep + xp); rz += p
                if fill: nT += 1; wins += p > 0
                real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if nf: cap = ep * (1.0 - capP * tg); nE += 1
        fill = nf if (tg != 0) else False
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nE, nS, nT, wins
S3, A3 = mk('3h')
for bo in (0, 1):
    for c_ in (0.04, 0.05):
        e, nE, nS, nT, w = comb2(o_, h_, l_, cc, 0.01, rp, rx, rn, S3, A3, 40.0, c_, bo); pr(f'{"stop sonrası blok" if bo else "blok yok"} stop %{c_ * 100:.0f}', met(e, ''))
        print(f'   ek giriş {nE}, stop {nS}, ek kapanan {nT}, kazanan %{w / max(nT, 1) * 100:.0f}')
