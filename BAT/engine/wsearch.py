import numpy as np, pandas as pd, json, time, sys
from vote import vote_pos
from eng import load
from lib import simR, cut
from met import metrics, SPLIT
D0 = load(5); X0 = np.load('comps.npy'); k0 = np.searchsorted(D0['t'], SPLIT); DI = cut(D0, SPLIT); XI = X0[:, :k0]
WV = [0.0, 0.5, 1.0, 2.0, 3.0]; ON = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]; OFFR = [0.0, 0.5]; KK = [1, 3]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def ev(w, on, offr, k):
    tg, _ = vote_pos(XI, np.array(w, np.float64), on, on * offr, k); m = metrics(DI, simR(DI, tg)); return m['net'] - 1.5 * m['p98'], m
best = None; rows = []; t0 = time.time()
for it in range(2500):
    if best is None or rng.random() < 0.35:
        w = [WV[rng.integers(5)] for _ in range(14)]; on = ON[rng.integers(6)]; offr = OFFR[rng.integers(2)]; k = KK[rng.integers(2)]
    else:
        w, on, offr, k = list(best[0]), best[1], best[2], best[3]; j = rng.integers(17)
        if j < 14: w[j] = WV[rng.integers(5)]
        elif j == 14: on = ON[rng.integers(6)]
        elif j == 15: offr = OFFR[rng.integers(2)]
        else: k = KK[rng.integers(2)]
    if sum(w) == 0: continue
    s, m = ev(w, on, offr, k); rows.append(dict(w=json.dumps(w), on=on, offr=offr, k=k, sc=s, net=m['net'], n=m['n'], pf=m['pf'], sharpe=m['sharpe'], p98=m['p98']))
    if best is None or s > best[4]: best = (w, on, offr, k, s)
df = pd.DataFrame(rows); df.to_pickle(f'wsearch_{sys.argv[1] if len(sys.argv) > 1 else 0}.pkl'); print(f'{time.time()-t0:.0f}s')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(df.sort_values('sc', ascending=False).head(8).to_string(index=False))
