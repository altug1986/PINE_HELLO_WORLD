import numpy as np, numba as nb, pandas as pd, json, time, sys, itertools
from scan import D, F, system, simR, metrics, SPLIT, TICK
from eng import dmi, htf_map
# V5.1 + POZ İÇİ ZARAR SINIRI + ÇIKIŞ SONRASI DÜZENLİ GERİ GİRİŞ + 3 SAATLİK OI + 2 SAATLİK BTC ADX FİLTRESİ
CFG = json.load(open('v51_cfg.json')); CFG['p'] = {int(k): v for k, v in CFG['p'].items()}
c = D['c']; t = D['t']; n = len(c)
# BTC 2 saatlik ADX ve DI farkı (önceki tamamlanmış 2s mum)
b = pd.read_csv('btc1m.csv.gz', usecols=['time', 'open', 'high', 'low', 'close']).to_numpy(float)
k2 = (b[:, 0] // 7200000).astype(np.int64); st_ = np.r_[0, np.flatnonzero(np.diff(k2)) + 1]; en_ = np.r_[st_[1:], len(b)]
bh = np.maximum.reduceat(b[:, 2], st_); bl = np.minimum.reduceat(b[:, 3], st_); bc = b[en_ - 1, 4]; bk = k2[st_]
pdi, mdi, badx = dmi(bh, bl, bc, 14, 14); mp = (t // 7200000).astype(np.int64)
ADX2 = np.nan_to_num(htf_map(badx, bk, mp)); DI2 = np.nan_to_num(htf_map(pdi - mdi, bk, mp))
oiP = np.r_[np.nan, D['oi'][:-1]]
OI3 = np.nan_to_num(np.r_[np.full(36, np.nan), (oiP[36:] / oiP[:-36] - 1) * 100])
PX3 = np.nan_to_num(np.r_[np.full(36, np.nan), (c[36:] / c[:-36] - 1) * 100])

@nb.njit(cache=True)
def layer(c, tg, ADX2, DI2, OI3, PX3, S, rmode, rprm, yA, xO, fm, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; lock = 0; lext = 0.0; lpx = 0.0; lbar = 0; prev = 0; nst = 0
    for i in range(n):
        t_ = tg[i]
        if t_ != prev:
            lock = 0; prev = t_
        want = t_
        if want != 0 and (fm == 1 or sh != want):          # filtreler: fm=0 sadece yeni girişte, fm=1 pozisyon içinde de
            if yA > 0 and ADX2[i] > yA and DI2[i] * want < 0: want = 0
            if want != 0 and xO > 0 and OI3[i] >= xO and PX3[i] * want < 0: want = 0
        if sh != 0 and want == sh and S < 50:
            if (c[i] / ep - 1) * sh * 100 <= -S:
                lock = sh; lext = c[i]; lpx = c[i]; lbar = i; want = 0; nst += 1
        if lock != 0 and want == lock:
            if lock > 0: lext = min(lext, c[i])
            else: lext = max(lext, c[i])
            ok = False
            if rmode == 1: ok = (c[i] / lext - 1) * lock * 100 >= rprm
            elif rmode == 2: ok = i - lbar >= rprm
            elif rmode == 3: ok = (c[i] - lpx) * lock >= 0
            if ok: lock = 0
            else: want = 0
        if want != sh:
            sh = want; ep = c[i] + tick * want
        out[i] = sh
    return out, nst

RM = [(0, 0), (1, 1), (1, 2), (1, 3), (2, 12), (2, 48), (2, 144), (3, 0)]
def run(tg, p):
    return layer(c, tg, ADX2, DI2, OI3, PX3, p['S'], p['rm'], p['rp'], p['yA'], p['xO'], p['fm'], TICK)
def mae(R):
    w = np.array([(np.min(D['l'][a + 1:x + 1]) / e - 1) * 100 if s > 0 else -(np.max(D['h'][a + 1:x + 1]) / e - 1) * 100
                  for a, x, s, e in zip(R['eb'], R['xb'], R['dir'], R['ep']) if x > a])
    return np.percentile(w, [10, 5, 1]).round(1)
if __name__ == '__main__':
    tg0 = system(CFG); k0 = np.searchsorted(t, SPLIT); km = np.searchsorted(t, np.int64(pd.Timestamp('2022-06-15').value // 10**6))
    def cutD(a, b_): return {k: (v[a:b_] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in D.items()}
    H = [cutD(0, km), cutD(km, k0)]
    def halves(tg):
        out = []
        for (a, b_), Dh in zip([(0, km), (km, k0)], H):
            m = metrics(Dh, simR(Dh, tg[a:b_])); out.append(m['net'] - 1.5 * m['p98'])
        return min(out), out
    s0 = halves(tg0); print('V5.1 ilk dönem yarı skorları', [round(x) for x in s0[1]], flush=True)
    seed = int(sys.argv[1]); rows = []; t0 = time.time()
    G = list(itertools.product([3.0, 4.0, 5.0, 6.0, 8.0, 99.0], range(len(RM)), [0.0, 20.0, 25.0, 30.0, 35.0], [0.0, 1.0, 2.0, 3.0, 5.0], [0, 1]))
    G = [g for g in G if not (g[0] >= 50 and g[1] > 0)]
    G = G[seed::2]
    for S, ri, yA, xO, fm in G:
        if yA == 0 and xO == 0 and fm == 1: continue
        p = dict(S=S, rm=RM[ri][0], rp=RM[ri][1], yA=yA, xO=xO, fm=fm); tg, nst = run(tg0, p); s, hs = halves(tg)
        rows.append(dict(**p, stop_say=nst, sc=s, a=hs[0], b=hs[1]))
    df = pd.DataFrame(rows); df.to_pickle(f'stopre_{seed}.pkl'); print(len(df), 'varyant', f'{time.time()-t0:.0f}s')
