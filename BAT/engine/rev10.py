import numpy as np, numba as nb, pandas as pd, json, time, sys
from vote import vote_pos
from eng import load, TICK
from lib import simR, cut
from met import metrics, SPLIT
# KONSORSİYUM + "TESTE KALINCA ISRAR ETME, DESTEKLE DOĞRU TARAFA DÖN" katmanı
# destek: BTC çoklu zaman dilimi (1s / 4s / 1g getirisi hareket yönünde), BAT OI (girişten beri artış), konsorsiyum skoru zayıflaması
D0 = load(5); X0 = np.load('comps.npy'); c0 = D0['c']; xc = D0['xc']
def ret(a, k): return np.r_[np.full(k, np.nan), a[k:] / a[:-k] - 1] * 100
BT = np.stack([np.nan_to_num(ret(xc, 12)), np.nan_to_num(ret(xc, 48)), np.nan_to_num(ret(xc, 288))])
OI = np.r_[np.nan, D0['oi'][:-1]]; ATR = np.nan_to_num(D0['atr1h'])
W6 = np.array([3, 3, 0.5, 0, 0, 1, 3, 3, 0.5, 1, 0.5, 0.5, 2, 0], float); DON = np.array([1] * 10 + [0] * 4, bool)

@nb.njit(cache=True)
def layer(c, tg, sc, BT, OI, ATR, thrK, mB, oiX, sThr, cSL, cBars, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; oe = np.nan; st = 0; cDir = 0; cPx = 0.0; cBar = 0; prev = 0
    for i in range(n):
        t_ = tg[i]; evt = t_ != prev; prev = t_; w = sh
        if evt:
            st = 0; w = t_
        elif st == 0:
            w = t_
            if sh != 0 and sh == t_ and ATR[i] > 0:
                adv = (c[i] / ep - 1) * (-sh) * 100
                if adv >= thrK * ATR[i]:
                    nb_ = 0
                    for k in range(3):
                        if BT[k, i] * (-sh) > 0: nb_ += 1
                    okO = oiX < -50 or ((not np.isnan(OI[i])) and (not np.isnan(oe)) and oe > 0 and (OI[i] / oe - 1) * 100 >= oiX)
                    okS = sThr < -50 or sc[i] * sh <= sThr
                    if nb_ >= mB and okO and okS:
                        w = -sh; st = 1; cDir = sh; cPx = c[i]; cBar = i
        else:
            g = (c[i] / cPx - 1) * (-cDir) * 100
            if g <= -cSL or i - cBar >= cBars:
                st = 0; w = t_
            else:
                w = -cDir
        if w != sh:
            sh = w; ep = c[i] + tick * w; oe = OI[i]
        out[i] = w
    return out

def build(D, X, BTs, OIs, ATRs, p):
    w = W6 * np.where(DON, p['sD'], p['sT'])
    if w.sum() == 0: w = W6.copy()
    tg, sc = vote_pos(X, w, p['on'], p['on'] * p['offr'], 1)
    if p['thrK'] >= 99: return tg
    return layer(D['c'], tg, sc, BTs, OIs, ATRs, p['thrK'], p['mB'], p['oiX'], p['sThr'], p['cSL'], p['cBars'], TICK)

SP = dict(on=[0.03, 0.05, 0.08, 0.1, 0.15], offr=[0.0, 0.5], thrK=[99, 2, 3, 4, 6, 8], mB=[0, 1, 2, 3], oiX=[-99, 0, 2, 5], sThr=[-99, -0.05, 0.0, 0.05],
          cSL=[1, 2, 3, 5], cBars=[48, 144, 288, 576], sD=[0.5, 1.0, 1.5], sT=[0.0, 0.5, 1.0, 1.5])   # 10 boyut
if __name__ == '__main__':
    k0 = np.searchsorted(D0['t'], SPLIT); MID = np.int64(pd.Timestamp('2022-06-15').value // 10**6); km = np.searchsorted(D0['t'], MID)
    DI = cut(D0, SPLIT); XI = X0[:, :k0]; BTI = BT[:, :k0]; OII = OI[:k0]; ATI = ATR[:k0]
    def halves(tg):
        out = []
        for a, b in [(0, km), (km, k0)]:
            Dh = {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == len(DI['t']) else v) for k, v in DI.items()}
            m = metrics(Dh, simR(Dh, tg[a:b])); out.append((m['net'] - 1.5 * m['p98'], m['ulcer']))
        return min(o[0] for o in out), out
    rng = np.random.default_rng(int(sys.argv[1])); best = None; rows = []; t0 = time.time()
    for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1200):
        if best is None or rng.random() < 0.35: p = {k: v[rng.integers(len(v))] for k, v in SP.items()}
        else:
            p = dict(best[0]); k = list(SP)[rng.integers(len(SP))]; p[k] = SP[k][rng.integers(len(SP[k]))]
        s, hs = halves(build(DI, XI, BTI, OII, ATI, p)); rows.append(dict(p=json.dumps(p), sc=s, a=hs[0][0], b=hs[1][0]))
        if best is None or s > best[1]: best = (p, s)
    df = pd.DataFrame(rows); df.to_pickle(f'rev10_{sys.argv[1]}.pkl'); print(f'{time.time()-t0:.0f}s')
    base = halves(build(DI, XI, BTI, OII, ATI, dict(on=0.05, offr=0.0, thrK=99, mB=0, oiX=-99, sThr=-99, cSL=1, cBars=48, sD=1.0, sT=1.0)))
    print('V5.0 IS yarı skorları', [round(x[0]) for x in base[1]], 'min', round(base[0]))
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200); print(df.sort_values('sc', ascending=False).drop_duplicates('p').head(5).to_string(index=False))
