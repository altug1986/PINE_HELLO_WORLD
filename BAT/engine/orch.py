import numpy as np, numba as nb, pandas as pd, itertools, json, time, sys
from zz import g_rev
from eng import load, TICK, CASH, COMM
from lib import simR, cut
from met import metrics, SPLIT

INST = [('EN KISA', 2, 2), ('KISA', 3, 3), ('ORTA', 5, 5), ('UZUN', 8, 8)]   # boyut 1: ufuk
WIN = [1, 7, 30, 90]                                                     # boyut 2: zaman penceresi (gün)
BPD = 288

@nb.njit(cache=True)
def shadow_cum(c, s, tick):
    # gölge işlemleri → çıkış barında yön bazlı kümülatif brüt kâr / zarar / adet (boyut 3: yön)
    n = len(c); GP = np.zeros((2, n)); GL = np.zeros((2, n)); CN = np.zeros((2, n)); prev = 0; ep = 0.0
    gp = np.zeros(2); gl = np.zeros(2); cn = np.zeros(2)
    for i in range(n):
        x = s[i]
        if x != prev:
            if prev != 0:
                px = c[i] - tick * prev; q = CASH / ep; p = (px - ep) * q * prev - COMM * q * (ep + px); d = 0 if prev > 0 else 1
                if p > 0: gp[d] += p
                else: gl[d] -= p
                cn[d] += 1
            ep = c[i] + tick * x; prev = x
        for d in range(2):
            GP[d, i] = gp[d]; GL[d, i] = gl[d]; CN[d, i] = cn[d]
    return GP, GL, CN

def roll(A, w):
    out = A.copy(); out[..., w:] = A[..., w:] - A[..., :-w]; return out

def scores(D, S, dirspec, minc, wts, clip=1.5):
    # her enstrüman için şu anki yönündeki (veya tüm) işlemlerin pencere log-PF'lerinin ağırlıklı toplamı
    n = len(D['c']); SC = np.zeros((len(S), n))
    for i, s in enumerate(S):
        GP, GL, CN = shadow_cum(D['c'], s, TICK); acc = np.zeros(n)
        for w, a in zip(WIN, wts):
            gp, gl, cn = roll(GP, w * BPD), roll(GL, w * BPD), roll(CN, w * BPD)
            if dirspec:
                d = np.where(s > 0, 0, 1); g = gp[d, np.arange(n)]; l = gl[d, np.arange(n)]; k = cn[d, np.arange(n)]
            else:
                g = gp.sum(0); l = gl.sum(0); k = cn.sum(0)
            lp = np.clip(np.log((g + 1.0) / (l + 1.0)), -clip, clip); lp[k < minc] = 0.0; acc += a * lp
        SC[i] = acc
    return SC

@nb.njit(cache=True)
def conductor(S, SC, on, off, k, pw):
    # şef: iyi çalan enstrümanların (skor>0) yön oyu; histerezis (on/off) + tempo (k bar teyit)  — boyut 4: tempo
    m, n = SC.shape; tg = np.zeros(n, np.int64); pos = 0; run = 0; last = 0
    for i in range(n):
        C = 0.0; tot = 0.0
        for j in range(m):
            w = SC[j, i]
            if w > 0:
                C += (w ** pw) * S[j, i]
            tot += abs(SC[j, i]) ** pw
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

WTS = {'eşit': [0.25, 0.25, 0.25, 0.25], 'kısa-ağır': [0.4, 0.3, 0.2, 0.1], 'uzun-ağır': [0.1, 0.2, 0.3, 0.4]}
if __name__ == '__main__':
    D0 = load(5); DI = cut(D0, SPLIT); S_I = np.stack([g_rev(DI['c'], u, d) for _, u, d in INST]); rows = []; t0 = time.time()
    for dirspec, minc, wn in itertools.product([0, 1], [3, 6], WTS):
        SC = scores(DI, S_I, dirspec, minc, WTS[wn])
        for on, r, k, pw in itertools.product([0.05, 0.1, 0.2, 0.3, 0.45], [0.0, 0.5], [1, 3, 6], [1, 2]):
            R = simR(DI, conductor(S_I, SC, on, r * on, k, pw)); m = metrics(DI, R)
            rows.append(dict(dirspec=dirspec, minc=minc, wts=wn, on=on, offr=r, k=k, pw=pw, **{kk: m[kk] for kk in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'upi', 'posM']}))
        print(dirspec, minc, wn, f'{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows); df['sc'] = df.net - 1.5 * df.p98; df.to_pickle('orch_IS.pkl')
    ok = df[(df.n >= 1300) & (df.n <= 1800)]
    print('IS kârlı oran', (df.net > 0).mean().round(2), '| 1300–1800 IS işlem aralığında', len(ok), 'kârlı', (ok.net > 0).mean().round(2))
    pd.set_option('display.width', 250); print(ok.sort_values('sc', ascending=False).head(10).to_string(index=False))
