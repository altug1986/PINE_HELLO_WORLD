import numpy as np, numba as nb, pandas as pd, itertools, json
from zz import g_rev
from eng import load, rma, trr, htf_map, TICK, CASH, COMM
from lib import simR, cut
from met import metrics, SPLIT
from scipy.ndimage import uniform_filter
# TAHMİNCİ 4B FİLTRE — Pine'a birebir: gölge 3/3 takip; her gölge dönüşünde hücre beklentisine göre TAKİP / TERS / FLAT
E24 = np.array([-5.8, -1.25, 1.45, 5.8]); EVR = np.array([0.88, 1.34]); E7 = np.array([-15.0, -4.3, 4.3, 15.0])

def features(D):
    c = D['c']; z = dict(np.load('d5.npz'))
    b = z['b60']; at = rma(trr(b[1], b[2], b[3], True), 14) / b[3] * 100
    ratio = at / pd.Series(at).rolling(720).median().to_numpy()           # 1S ATR% / son 720 saatlik medyan
    volr = htf_map(ratio, z['b60_k'], z['b60_map'])                        # önceki tamamlanmış 1S mum
    k = len(c)
    e24 = np.r_[np.full(288, np.nan), (c[288:] / c[:-288] - 1) * 100]
    e7 = np.r_[np.full(2016, np.nan), (c[2016:] / c[:-2016] - 1) * 100]
    return e24, volr[:k], e7

@nb.njit(cache=True)
def bucket(x, edges):
    if np.isnan(x): return len(edges) // 2
    b = 0
    for e in edges:
        if x >= e: b += 1
    return b

@nb.njit(cache=True)
def run_exact(c, s, e24, volr, e7, kk, th, inv, tick):
    n = len(c); tg = np.zeros(n, np.int64)
    Sf = np.zeros(4 * 1024); Si = np.zeros(4 * 1024); N = np.zeros(4 * 1024)
    prev = 0; real = 0; ep = 0.0; q = 0.0; cell = np.zeros(4, np.int64); acts = np.zeros(n, np.int64); na = 0
    for i in range(n):
        x = s[i]
        if x != prev:
            if prev != 0:   # gölge işlemi kapat → hücrelerine öğret
                px = c[i] - tick * prev; p = (px - ep) * q * prev - COMM * q * (ep + px); r = -p - 2 * COMM * q * (ep + px) - 4 * tick * q
                for lv in range(4):
                    Sf[lv * 1024 + cell[lv]] += p; Si[lv * 1024 + cell[lv]] += r; N[lv * 1024 + cell[lv]] += 1.0
            if x != 0:      # yeni gölge işlem: bağlam → hücre → tahmin → karar
                b1 = bucket(e24[i] * x, E24); b2 = bucket(volr[i], EVR); b3 = bucket(e7[i] * x, E7)
                k0 = 1 if x > 0 else 0; cell[0] = k0; cell[1] = k0 * 8 + b1; cell[2] = cell[1] * 8 + b2; cell[3] = cell[2] * 8 + b3
                ef = 0.0; ei = 0.0
                for lv in range(4):
                    j = lv * 1024 + cell[lv]; ef = (Sf[j] + kk * ef) / (N[j] + kk); ei = (Si[j] + kk * ei) / (N[j] + kk)
                a = 0
                if ef >= th and ef >= ei: a = 1
                elif inv == 1 and ei >= th: a = -1
                real = a * x; acts[na] = a; na += 1
                ep = c[i] + tick * x; q = CASH / c[i]
            else:
                real = 0
            prev = x
        tg[i] = real
    return tg, acts[:na]

def system(D, F, kk=20.0, th=2.0, inv=1, u=3.0):
    s = g_rev(D['c'], u, u); return run_exact(D['c'], s, F[0], F[1], F[2], kk, th, inv, TICK)[0]

if __name__ == '__main__':
    D0 = load(5); F0 = features(D0); k0 = np.searchsorted(D0['t'], SPLIT); DI = cut(D0, SPLIT); FI = tuple(f[:k0] for f in F0)
    G = dict(kk=[5, 10, 20, 50, 100], th=[0, 0.5, 1, 1.5, 2, 2.5, 3], inv=[0, 1])
    A = np.zeros([len(v) for v in G.values()]); N = np.zeros_like(A); NT = np.zeros_like(A)
    for idx in itertools.product(*[range(len(v)) for v in G.values()]):
        p = {k: G[k][i] for k, i in zip(G, idx)}; m = metrics(DI, simR(DI, system(DI, FI, **p))); A[idx] = m['net'] - 1.5 * m['p98']; N[idx] = m['net']; NT[idx] = m['n']
    Sm = uniform_filter(A, size=3, mode='nearest'); i1 = np.unravel_index(A.argmax(), A.shape); i2 = np.unravel_index(Sm.argmax(), A.shape)
    for lab, i in [('tepe', i1), ('plato merkezi', i2)]:
        print(lab, {k: G[k][j] for k, j in zip(G, i)}, 'IS net', N[i], 'işlem', NT[i], 'skor', round(A[i]), 'komşu', round(Sm[i]))
    print('IS kârlı oran', (N > 0).mean().round(2)); print(pd.DataFrame(N[:, :, 1], index=G['kk'], columns=G['th']).round(0).to_string())
    best = {k: G[k][j] for k, j in zip(G, i2)}; json.dump(best, open('onlx.json', 'w'))
