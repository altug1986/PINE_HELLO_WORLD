import numpy as np, pandas as pd, itertools, sys, time
import base56b, v55
from base56b import A
from v55 import evalp, c, D, n
lo, hi = D['l'], D['h']
def don(N):
    ll = pd.Series(lo).rolling(N).min().shift(1).to_numpy(); hh = pd.Series(hi).rolling(N).max().shift(1).to_numpy()
    return np.where(c < ll, -1, np.where(c > hh, 1, 0)).astype(np.int64)
DON = {N: don(N) for N in (12, 24, 48, 96, 288)}
if __name__ == '__main__':
    G = list(itertools.product([12, 24, 48, 96, 288], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3], [2.0, 3.0, 5.0], [48, 144, 288, 576]))[int(sys.argv[1])::2]
    rows = []; t0 = time.time()
    for NL, x, act, cS, N in G:
        v55.ZC = DON[NL]; p = dict(A, zK1=0, zK2=10**7, zX=x, zM=99.0, zC=1, zAct=act, cS=cS, N=N); rows.append(dict(NL=NL, x=x, act=act, cS=cS, N=N, **evalp(p)))
    pd.DataFrame(rows).to_pickle(f'pa_{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
