import numpy as np, numba as nb
from orch3 import *
@nb.njit(cache=True)
def inc(c, S, wb, minc, on, off, k, tick):
    # Pine'daki işaretçi (pointer) mantığının birebir kopyası
    m, n = S.shape; MX = 20000
    XB = np.zeros((m, MX), np.int64); PL = np.zeros((m, MX)); NT = np.zeros(m, np.int64)
    PTR = np.zeros((m, 4), np.int64); GP = np.zeros((m, 4)); GL = np.zeros((m, 4)); CN = np.zeros((m, 4))
    EP = np.zeros(m); PV = np.zeros(m, np.int64); tg = np.zeros(n, np.int64); pos = 0; run = 0; last = 0
    for i in range(n):
        SC = np.zeros(m)
        for j in range(m):
            s = S[j, i]
            if s != PV[j]:
                if PV[j] != 0:
                    px = c[i] - tick * PV[j]; q = 500.0 / EP[j]; p = (px - EP[j]) * q * PV[j] - 0.0004 * q * (EP[j] + px)
                    XB[j, NT[j]] = i; PL[j, NT[j]] = p; NT[j] += 1
                    for w in range(4):
                        if p > 0: GP[j, w] += p
                        else: GL[j, w] -= p
                        CN[j, w] += 1
                EP[j] = c[i] + tick * s; PV[j] = s
            sc = 0.0
            for w in range(4):
                lim = i - wb[w]; p_ = PTR[j, w]
                while p_ < NT[j] and XB[j, p_] <= lim:
                    pv = PL[j, p_]
                    if pv > 0: GP[j, w] -= pv
                    else: GL[j, w] += pv
                    CN[j, w] -= 1; p_ += 1
                PTR[j, w] = p_
                if CN[j, w] >= minc:
                    lp = np.log((GP[j, w] + 1.0) / (GL[j, w] + 1.0)); lp = min(max(lp, -1.5), 1.5); sc += 0.25 * lp
            SC[j] = sc
        C = 0.0; tot = 0.0
        for j in range(m):
            if SC[j] > 0: C += SC[j] * S[j, i]
            tot += abs(SC[j])
        C = C / tot if tot > 0 else 0.0
        want = pos
        if pos != 0 and C * pos < off: want = 0
        if C >= on: want = 1
        elif C <= -on: want = -1
        if want != 0 and want != pos:
            run = run + 1 if want == last else 1
            last = want
            if run < k: want = pos if (pos == 0 or C * pos >= off) else 0
        else:
            run = 0; last = 0
        pos = want; tg[i] = pos
    return tg
D0 = load(5); F0 = feats(D0); S = inst_set(D0, F0, BASE8)
SC = scores(D0, S, 0, 6, WTS['eşit']); a = conductor(S, SC, 0.10, 0.0, 6, 1)
b = inc(D0['c'], S, np.array([288, 2016, 8640, 25920]), 6, 0.10, 0.0, 6, TICK)
print('vektör vs işaretçi uyuşmazlık bar:', int((a != b).sum()), '| işlem', len(simR(D0, a)['pnl']), len(simR(D0, b)['pnl']))
