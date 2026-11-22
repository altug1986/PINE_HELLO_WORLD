import numpy as np, pandas as pd, itertools, json, time
import v55
from v55 import evalp
from rg55 import vote2, AGREE, ADX60, X15, A
from v54 import CFG
from scan import comp
from vote import vote_pos
rows = []; t0 = time.time()
for x11 in [2.0, 2.5]:
    p11 = dict(CFG['p'][11]); p11['x'] = x11
    X = np.vstack([np.stack([comp(j, p11 if j == 11 else CFG['p'][j]) for j in range(1, 15)]), X15[None, :]]); _, sc = vote_pos(X, np.array(CFG['w'] + [1.0], float), CFG['on'], CFG['off'], 1)
    for on, onT, aT in itertools.product([0.04, 0.05, 0.06, 0.08], [0.0, 0.10, 0.12, 0.15], [25.0, 30.0]):
        ot = on if onT == 0 else onT
        v55.TG = vote2(sc, ADX60, AGREE, 0.0, aT, on, on, ot, on, 0.0); rows.append(dict(x11=x11, on=on, onT=ot, aT=aT, **evalp(A)))
d = pd.DataFrame(rows).drop_duplicates(subset=['x11', 'on', 'onT', 'net']); d.to_pickle('rg56.pkl'); pd.set_option('display.width', 250); print(len(d), f'{time.time()-t0:.0f}s')
M = ['x11', 'on', 'onT', 'aT', 'sc', 'net', 'n', 'pf', 'calmar', 'p98', 'mddc', 'mddi', 'IS', 'OOS', 'L12', 'L24', 'bmin', 'bstd', 'bneg']
print(d.sort_values('sc', ascending=False).head(10)[M].to_string(index=False))
print(d.sort_values('net', ascending=False).head(5)[M].to_string(index=False))
