import itertools, pandas as pd, time, sys
from v55 import evalp
B = dict(H=576, xH=-2.0, rH=4.0, aH=20.0); rows = []; t0 = time.time()
G = list(itertools.product([7.0, 7.5, 8.0], [(0, 0, 0.0), (50, 0, 5.0), (50, 20, 5.0), (60, 20, 5.0), (60, 0, 6.0)], [0, 0.1, 0.2], [0, 0.05, 0.1], [5.0, 99.0], [0, 48]))[int(sys.argv[1])::2]
for S, (aT, dT, sT), nm, vm, L, hm in G:
    p = dict(B, S=S, aT=aT, dT=dT, sT=sT, nrMin=nm, vrMin=vm, L=L, hm=hm); rows.append(dict(**p, **evalp(p)))
pd.DataFrame(rows).to_pickle(f's55_D{sys.argv[1]}.pkl'); print(len(rows), f'{time.time()-t0:.0f}s')
