import numpy as np, numba as nb, pandas as pd, itertools, json, time, sys
from vote import vote_pos
from eng import load, TICK
from lib import simR, cut
from met import metrics, SPLIT
# ZARAR %5'İ GEÇİNCE "TERS YÖNE GEÇMEK İÇİN FIRSAT KOLLA"
# tip 0 hemen | 1 devam (silahlanınca +x% daha ters giderse) | 2 tepki satışı/alımı (ters uçtan r% geri tepince karşı yöne)
# tip 3 konsorsiyum skoru ters yöne dönerse | 4 BTC 1 saatlik getirisi hareket yönündeyse
# bekleme: 0 = pozisyonu koru, 1 = flat bekle ; fırsat wBars içinde gelmezse normale dön
D0 = load(5); X0 = np.load('comps.npy'); W6 = np.array([3, 3, 0.5, 0, 0, 1, 3, 3, 0.5, 1, 0.5, 0.5, 2, 0], float)
xc = D0['xc']; B1 = np.nan_to_num(np.r_[np.full(12, np.nan), xc[12:] / xc[:-12] - 1] * 100)

@nb.njit(cache=True)
def opp(c, tg, sc, b1, arm, typ, prm, wait, wBars, cSL, cBars, on, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; st = 0; aDir = 0; aBar = 0; aPx = 0.0; ext = 0.0; cPx = 0.0; cBar = 0; prev = 0
    for i in range(n):
        t_ = tg[i]; evt = t_ != prev; prev = t_; w = sh
        if evt:
            st = 0; w = t_
        elif st == 0:
            w = t_
            if sh != 0 and sh == t_ and (c[i] / ep - 1) * sh * 100 <= -arm:
                st = 1; aDir = sh; aBar = i; aPx = c[i]; ext = c[i]
                w = sh if wait == 0 else 0
        if st == 1 and not evt:
            mv = -aDir                                     # hareketin yönü (pozisyonumuzun tersi)
            if mv > 0: ext = max(ext, c[i])
            else: ext = min(ext, c[i])
            go = False
            if typ == 0: go = True
            elif typ == 1: go = (c[i] / aPx - 1) * mv * 100 >= prm
            elif typ == 2: go = (c[i] / ext - 1) * (-mv) * 100 >= prm   # hareketin tersine r% tepki → karşı yöne gir (tepkiyi sat/al)
            elif typ == 3: go = sc[i] * mv >= on
            else: go = b1[i] * mv >= prm
            if go:
                st = 2; w = mv; cPx = c[i]; cBar = i
            elif i - aBar >= wBars:
                st = 0; w = t_
            else:
                w = aDir if wait == 0 else 0
        elif st == 2 and not evt:
            g = (c[i] / cPx - 1) * (-aDir) * 100
            if g <= -cSL or i - cBar >= cBars:
                st = 0; w = t_
            else:
                w = -aDir
        if w != sh:
            sh = w; ep = c[i] + tick * w
        out[i] = w
    return out

if __name__ == '__main__':
    k0 = np.searchsorted(D0['t'], SPLIT); MID = np.int64(pd.Timestamp('2022-06-15').value // 10**6); km = np.searchsorted(D0['t'], MID)
    DI = cut(D0, SPLIT); XI = X0[:, :k0]; tgI, scI = vote_pos(XI, W6, 0.05, 0.0, 1); bI = B1[:k0]
    def halves(tg):
        out = []
        for a, b in [(0, km), (km, k0)]:
            Dh = {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == len(DI['t']) else v) for k, v in DI.items()}
            m = metrics(Dh, simR(Dh, tg[a:b])); out.append(m['net'] - 1.5 * m['p98'])
        return min(out), out
    b0 = halves(tgI); print('V5.0 ilk dönem yarı skorları', [round(x) for x in b0[1]]); rows = []; t0 = time.time()
    PRM = {0: [0], 1: [1, 2, 4], 2: [1, 2, 3], 3: [0], 4: [0.3, 0.6, 1.0]}
    for typ in range(5):
        for prm, wait, wB, cSL, cB in itertools.product(PRM[typ], [0, 1], [144, 576], [2, 3, 5], [144, 288, 576]):
            tg = opp(DI['c'], tgI, scI, bI, 5.0, typ, prm, wait, wB, cSL, cB, 0.05, TICK); s, hs = halves(tg)
            rows.append(dict(tip=['hemen', 'devam', 'tepki', 'skor', 'BTC'][typ], prm=prm, bekle=['koru', 'flat'][wait], wBars=wB, cSL=cSL, cBars=cB, sc=s, a=hs[0], b=hs[1]))
    df = pd.DataFrame(rows); df.to_pickle('opp_IS.pkl'); print(f'{time.time()-t0:.0f}s', len(df))
    print('iki yarıda da V5.0 üstü oran', ((df.a > b0[1][0]) & (df.b > b0[1][1])).mean().round(3))
    print(df.groupby('tip').apply(lambda g: pd.Series(dict(en_iyi_min=round(g.sc.max()), iki_yarı_üstü=int(((g.a > b0[1][0]) & (g.b > b0[1][1])).sum()))), include_groups=False).to_string())
    pd.set_option('display.width', 250); print(df.sort_values('sc', ascending=False).head(6).round(0).to_string(index=False))
