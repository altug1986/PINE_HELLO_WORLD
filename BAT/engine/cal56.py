import numpy as np, pandas as pd, itertools, time
import v55
from v55 import evalp
from val56 import SCs, A
from rg55 import vote2, AGREE, ADX60
rows = []; t0 = time.time()
for x11, on, onT, aT, onM in itertools.product([2.0, 2.25, 2.5], [0.07, 0.08, 0.09], [0.10, 0.12, 0.15, 0.20], [20.0, 25.0, 30.0], [None, 0.10, 0.12, 0.15]):
    om = on if onM is None else onM
    v55.TG = vote2(SCs[x11], ADX60, AGREE, 0.0, aT, on, on, onT, om, 0.0); e = evalp(A); rows.append(dict(x11=x11, on=on, onT=onT, aT=aT, onM=om, **e))
d = pd.DataFrame(rows); d.to_pickle('cal56.pkl'); print(len(d), f'{time.time()-t0:.0f}s')
