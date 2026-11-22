import itertools, pandas as pd, time, sys
from v55 import evalp
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
B0 = {} if sys.argv[1] == 'base' else A; rows = []; t0 = time.time()
for tf, tA, tD, tS in itertools.product(['a15', 'a30', 'a60', 'a240', 'x60', 'x240'], [20, 25, 30, 35, 40, 50], [0, 1], [4.0, 4.5, 5.0, 5.5, 6.0]):
    p = dict(B0, tf=tf, tA=tA, tD=tD, tS=tS); rows.append(dict(tf=tf, tA=tA, tD=tD, tS=tS, **evalp(p)))
pd.DataFrame(rows).to_pickle(f'ts55_{sys.argv[1]}.pkl'); print(sys.argv[1], len(rows), f'{time.time()-t0:.0f}s')
