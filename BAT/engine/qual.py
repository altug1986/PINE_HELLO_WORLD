import numpy as np, numba as nb, pandas as pd, json, time, sys, itertools
from scan import D, F, comp, simR, metrics, SPLIT, TICK
from vote import vote_pos
# V5.2 + KALİTELİ EK GİRİŞLER / DÖNÜŞLER
CFG = json.load(open('v51_cfg.json')); CFG['p'] = {int(k): v for k, v in CFG['p'].items()}
c = D['c']; t = D['t']; n = len(c); xc = D['xc']
X = np.stack([comp(j, CFG['p'][j]) for j in range(1, 15)]); POS, SC = vote_pos(X, np.array(CFG['w'], float), CFG['on'], CFG['off'], 1)
B4 = np.nan_to_num(np.r_[np.full(48, np.nan), (xc[48:] / xc[:-48] - 1) * 100])

@nb.njit(cache=True)
def layer(c, pos, sc, X, B4, S, reArm, thRe, y, bconf, fN, fSL, m3, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; lock = 0; lext = 0.0; lpx = 0.0; prevT = 0
    fol = 0; fPx = 0.0; fBar = 0; nf = 0; n3 = 0
    for i in range(n):
        t_ = pos[i]
        if t_ != prevT:
            lock = 0; fol = 0; prevT = t_
        want = t_
        if fol != 0:                                           # M2: stop sonrası trend takip pozisyonu
            g = (c[i] / fPx - 1) * fol * 100
            if g <= -fSL or i - fBar >= fN:
                fol = 0; lext = c[i]; want = 0 if lock != 0 else t_
            else:
                want = fol
        else:
            if want == 0 and lock == 0 and m3 > 0:            # M3: konsorsiyum flatken kaliteli bileşen girişi
                d1 = X[0, i]
                if d1 != 0:
                    if m3 == 1: want = d1
                    elif m3 == 2 and (X[2, i] == d1 or X[1, i] == d1): want = d1
                    elif m3 == 3 and sc[i] * d1 > 0: want = d1
                    if want != 0: n3 += 1
            if sh != 0 and want == sh and S < 50:
                if (c[i] / ep - 1) * sh * 100 <= -S:
                    lock = sh; lext = c[i]; lpx = c[i]; want = 0
            if lock != 0 and want == lock:
                if lock > 0: lext = min(lext, c[i])
                else: lext = max(lext, c[i])
                if (c[i] / lext - 1) * lock * 100 >= reArm or (thRe < 50 and sc[i] * lock >= thRe):
                    lock = 0
                elif y < 50 and (c[i] / lpx - 1) * (-lock) * 100 >= y and (bconf == 0 or B4[i] * (-lock) > 0):
                    fol = -lock; fPx = c[i]; fBar = i; want = fol; nf += 1
                else:
                    want = 0
        if want != sh:
            sh = want; ep = c[i] + tick * want
        out[i] = sh
    return out, nf, n3

def run(p):
    return layer(c, POS, SC, X, B4, p['S'], p['reArm'], p['thRe'], p['y'], p['bc'], p['fN'], p['fSL'], p['m3'], TICK)
BASE = dict(S=8.0, reArm=3.0, thRe=99.0, y=99.0, bc=0, fN=144, fSL=3.0, m3=0)
if __name__ == '__main__':
    k0 = np.searchsorted(t, SPLIT); km = np.searchsorted(t, np.int64(pd.Timestamp('2022-06-15').value // 10**6))
    def cutD(a, b_): return {k: (v[a:b_] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in D.items()}
    H = [cutD(0, km), cutD(km, k0)]
    def halves(tg):
        out = []
        for (a, b_), Dh in zip([(0, km), (km, k0)], H):
            m = metrics(Dh, simR(Dh, tg[a:b_])); out.append(m['net'] - 1.5 * m['p98'])
        return min(out), out
    s0 = halves(run(BASE)[0]); print('V5.2 ilk dönem yarı skorları', [round(x) for x in s0[1]], flush=True)
    seed = int(sys.argv[1]); rows = []; t0 = time.time()
    G = list(itertools.product([99.0, 0.2, 0.3, 0.5], [99.0, 2.0, 4.0, 6.0], [0, 1], [48, 144, 288], [2.0, 3.0, 5.0], [0, 1, 2, 3]))
    G = [g for g in G if not (g[1] >= 50 and (g[2] or g[3] != 144 or g[4] != 3.0))][seed::2]
    for thRe, y, bc, fN, fSL, m3 in G:
        p = dict(BASE, thRe=thRe, y=y, bc=bc, fN=fN, fSL=fSL, m3=m3); tg, nf, n3 = run(p); s, hs = halves(tg)
        rows.append(dict(thRe=thRe, y=y, bc=bc, fN=fN, fSL=fSL, m3=m3, takip=nf, ek=n3, sc=s, a=hs[0], b=hs[1]))
    df = pd.DataFrame(rows); df.to_pickle(f'qual_{seed}.pkl'); print(len(df), 'varyant', f'{time.time()-t0:.0f}s')
