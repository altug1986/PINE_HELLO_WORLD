import numpy as np, numba as nb, pandas as pd, itertools, sys, time
import v55
from v55 import evalp, n
from v54 import X0, CFG, comp15
from vote import vote_pos
from tr1h import bat_e, btc_e, ADX60
X15 = comp15(1.5, 4.0); XX = np.vstack([X0, X15[None, :]]); WW = np.array(CFG['w'] + [1.0], float)
TG0, SC = vote_pos(XX, WW, CFG['on'], CFG['off'], 1); assert (TG0 == v55.TG).all()
bt = np.nan_to_num(np.sign(bat_e(5) - bat_e(17))).astype(np.int64); xt = np.nan_to_num(np.sign(btc_e(5) - btc_e(20))).astype(np.int64)
AGREE = np.where(bt == xt, bt, 0)
@nb.njit(cache=True)
def vote2(sc, adx, ag, aR, aT, onR, onN, onT, onM, off):
    n = len(sc); tg = np.zeros(n, np.int64); pos = 0
    for i in range(n):
        s = sc[i]; want = pos
        if pos != 0 and s * pos < off: want = 0
        if adx[i] < aR: oL = onR; oS = onR
        elif ag[i] == 0: oL = onM; oS = onM
        else:
            oL = onT if (ag[i] == -1 and adx[i] >= aT) else onN
            oS = onT if (ag[i] == 1 and adx[i] >= aT) else onN
        if s >= oL: want = 1
        elif s <= -oS: want = -1
        pos = want; tg[i] = pos
    return tg
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
if __name__ == '__main__':
    v55.TG = vote2(SC, ADX60, AGREE, 0.0, 999.0, 0.08, 0.08, 0.08, 0.08, 0.0); e = evalp(A); print('regresyon V5.5:', e['net'], e['L12'])
    G = list(itertools.product([15.0, 20.0, 25.0], [0.04, 0.05, 0.06, 0.08], [0.08, 0.10], [0.08, 0.10, 0.12, 0.15, 0.20], [0.08, 0.10, 0.12, 0.15], [25.0, 30.0]))[int(sys.argv[1])::2]
    rows = []; t0 = time.time()
    for aR, onR, onN, onT, onM, aT in G:
        v55.TG = vote2(SC, ADX60, AGREE, aR, aT, onR, onN, onT, onM, 0.0); rows.append(dict(aR=aR, onR=onR, onN=onN, onT=onT, onM=onM, aT=aT, **evalp(A)))
    pd.DataFrame(rows).to_pickle(f'rg55_{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
