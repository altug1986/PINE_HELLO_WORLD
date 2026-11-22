import numpy as np, pandas as pd, itertools, sys, time
import base56b, v55
from base56b import A
from v55 import evalp, D15
from tr1h import bat_e, btc_e, ADX60
from rg55 import AGREE
bt = np.nan_to_num(np.sign(bat_e(5) - bat_e(17))).astype(np.int64); xt = np.nan_to_num(np.sign(btc_e(5) - btc_e(20))).astype(np.int64)
ZCS = dict(yok=np.zeros(len(bt), np.int64), ortak1s=AGREE, ortak1s_adx25=np.where(ADX60 >= 25, AGREE, 0).astype(np.int64), bat1s=bt, btc1s=xt,
           di15=np.where(np.abs(D15) >= 10, np.sign(D15), 0).astype(np.int64))
if __name__ == '__main__':
    G = []
    for k2, x, m, zc in itertools.product([30, 50, 80], [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], [0.5, 1.0, 99.0], list(ZCS)):
        G.append(dict(zK2=k2, zX=x, zM=m, zc=zc, zAct=1))
        for act, cS, N in itertools.product([2, 3], [3.0, 5.0], [144, 288]): G.append(dict(zK2=k2, zX=x, zM=m, zc=zc, zAct=act, cS=cS, N=N))
    G = G[int(sys.argv[1])::2]; rows = []; t0 = time.time()
    for g in G:
        v55.ZC = ZCS[g['zc']]; p = dict(A, zK1=10, zK2=g['zK2'], zX=g['zX'], zM=g['zM'], zC=0 if g['zc'] == 'yok' else 1, zAct=g['zAct'])
        if 'cS' in g: p.update(cS=g['cS'], N=g['N'])
        rows.append(dict(**g, **evalp(p)))
    pd.DataFrame(rows).to_pickle(f'zone_{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
