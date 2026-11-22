import numpy as np, pandas as pd, sys, time
from v55 import evalp
rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
SP = dict(H=[0, 432, 504, 576, 648, 720], xH=[-3.0, -2.0, -1.0, 0.0], rH=[2.0, 4.0, 99.0], aT=[0, 45, 50, 55, 60], dT=[0, 10, 20], sT=[4.0, 5.0, 6.0],
          nrMin=[0, 0.1, 0.2, 0.3], vrMin=[0, 0.05, 0.1], hm=[0, 16, 32, 48], S=[6.5, 7.0, 7.5], rA=[3.0, 4.0, 5.0], L=[5.0, 99.0], a=[35.0, 40.0, 45.0])
while len(rows) < int(sys.argv[2]):
    p = {k: v[rng.integers(len(v))] for k, v in SP.items()}
    rows.append(dict(**p, **evalp(p)))
d = pd.DataFrame(rows); d.to_pickle(f's55_C{sys.argv[1]}.pkl'); print(len(d), f'{time.time()-t0:.0f}s')
