import numpy as np, pandas as pd, sys, time
from adxrev import *
if __name__ == '__main__':
    tg0 = system(CFG); b = metrics(D, simR(D, tg0)); rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
    SP = dict(L1=[2, 3, 4, 5, 6], a1=[18, 20, 23, 26, 30, 35], b1=[18, 22, 25, 30, 35], L2=[5, 6, 8, 10, 12], a2=[12, 15, 17, 20, 25], b2=[10, 15, 18, 22],
              cSL=[1, 2, 3, 5, 99], cBars=[48, 144, 288, 576, 10**9], re=[0, 1])
    while time.time() - t0 < float(sys.argv[2]):
        r = {k: float(v[rng.integers(len(v))]) for k, v in SP.items()}; r['cBars'] = int(r['cBars']); r['re'] = int(r['re'])
        tg, nf = run_rule(tg0, r); m = metrics(D, simR(D, tg)); rows.append(dict(**r, dönüş=nf, **{k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'p98', 'ulcer', 'IS', 'OOS', 'last12']}))
    df = pd.DataFrame(rows); df.to_pickle(f'adxscan_{sys.argv[1]}.pkl'); print(len(df), 'varyant')
