import numpy as np, numba as nb, pandas as pd, itertools, json, time, math
from eng import load, TICK
from lib import simR, cut
from met import metrics, SPLIT
from scipy.ndimage import uniform_filter
# KONSORSİYUM: 14 bileşenin pozisyonları → çevrimiçi ridge regresyon → önümüzdeki H barın beklenen getirisi (%)
# Öğrenme yalnızca geçmişten: saatlik örnek (x[i−H], getiri i−H→i), günlük (UTC 00:00) yeniden çözüm.
# Olasılık: P(yukarı) = Φ(E / σ), σ = artık standart sapması (çevrimiçi). Giriş/çıkış maliyet bilinçli eşiklerle.

@nb.njit(cache=True)
def consortium(c, t, X, H, lam, hl, on, off, minS):
    K, n = X.shape; K1 = K + 1
    A = np.zeros((K1, K1)); b = np.zeros(K1); w = np.zeros(K1); x = np.zeros(K1); nS = 0.0
    d = math.exp(-1.0 / hl) if hl > 0 else 1.0
    yy = 0.0; ysum = 0.0                       # σ için (ağırlıklı)
    tg = np.zeros(n, np.int64); E = np.zeros(n); S = np.zeros(n); pos = 0; ready = False
    for i in range(n):
        if (t[i] + 300000) % 3600000 == 0 and i >= H:
            j = i - H; x[0] = 1.0
            for k in range(K): x[k + 1] = X[k, j]
            y = (c[i] / c[j] - 1.0) * 100.0
            for r in range(K1):
                b[r] = b[r] * d + x[r] * y
                for q in range(K1): A[r, q] = A[r, q] * d + x[r] * x[q]
            nS = nS * d + 1.0; yy = yy * d + y * y; ysum = ysum * d + y
        if (t[i] + 300000) % 86400000 == 0 and nS >= minS:
            M = A.copy()
            for r in range(K1): M[r, r] += lam
            w = np.linalg.solve(M, b); ready = True
        e = w[0]
        for k in range(K): e += w[k + 1] * X[k, i]
        E[i] = e; S[i] = math.sqrt(max(yy / nS - (ysum / nS) ** 2, 1e-12)) if nS > 1 else 0.0
        if ready:
            if pos == 0:
                if e >= on: pos = 1
                elif e <= -on: pos = -1
            elif pos == 1:
                if e <= -on: pos = -1
                elif e < off: pos = 0
            else:
                if e >= on: pos = 1
                elif e > -off: pos = 0
        tg[i] = pos
    return tg, E, S, w

if __name__ == '__main__':
    D0 = load(5); X0 = np.load('comps.npy'); k0 = np.searchsorted(D0['t'], SPLIT); DI = cut(D0, SPLIT); XI = X0[:, :k0]
    G = dict(H=[48, 144, 288], lam=[10.0, 100.0, 1000.0, 10000.0], hl=[0.0, 8760.0], on=[0.1, 0.2, 0.3, 0.5, 0.8], offr=[0.0, 0.5])
    rows = []; t0 = time.time()
    for H, lam, hl, on, offr in itertools.product(*G.values()):
        tg, E, S, w = consortium(DI['c'], DI['t'], XI, H, lam, hl, on, on * offr, 720.0)
        m = metrics(DI, simR(DI, tg)); rows.append(dict(H=H, lam=lam, hl=hl, on=on, offr=offr, **{k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'mddc', 'upi', 'posM']}))
    df = pd.DataFrame(rows); df['sc'] = df.net - 1.5 * df.p98; df.to_pickle('cons_IS.pkl'); print(f'{time.time()-t0:.0f}s', len(df))
    A = df.sc.to_numpy().reshape([len(v) for v in G.values()]); Sm = uniform_filter(A, size=3, mode='nearest')
    i1 = np.unravel_index(A.argmax(), A.shape); i2 = np.unravel_index(Sm.argmax(), A.shape)
    for lab, i in [('tepe', i1), ('plato merkezi', i2)]:
        p = {k: G[k][j] for k, j in zip(G, i)}; r = df[(df.H == p['H']) & (df.lam == p['lam']) & (df.hl == p['hl']) & (df.on == p['on']) & (df.offr == p['offr'])].iloc[0]
        print(lab, p, {k: r[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'upi', 'posM']}, 'komşu skor', round(Sm[i]))
    print('IS kârlı oran', (df.net > 0).mean().round(2)); pd.set_option('display.width', 250)
    print(df.sort_values('sc', ascending=False).head(10).to_string(index=False))
    json.dump({k: G[k][j] for k, j in zip(G, i2)}, open('cons_best.json', 'w'))
