import numpy as np, pandas as pd, sys, time, pickle
from eng import *
from met import metrics, monkey, regimes
tf = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
D = load(tf); RG = regimes(D)
Pv = P()
T = [('MOTOR A (V2)', [0,0,1,0,0,0,0,0,1]), ('V3 (A veya B)', [0,0,1,0,0,0,0,0,2]), ('ÇARPIŞMA', [0,0,1,0,0,0,0,0,0]),
     ('RSI UÇ', [0,1,0,0,0,0,0,0,0]), ('ORTA REJİM', [0,0,0,0,1,0,0,0,0]), ('BTC 4H', [0,0,0,1,0,0,0,0,0]),
     ('LİKİDASYON', [1,0,0,0,0,0,0,0,0]), ('ZİNCİR (durum mak. yok)', [1,1,1,1,1,0,0,0,0]), ('BAZ-0 V11.7 tam', list(FULL))]
rows = []
for lab, f in T:
    t0 = time.time(); R = go(D, Pv, np.array(f, np.int64)); m = metrics(D, R, lab)
    mk = monkey(D, R, RG, N); m.update({k: round(v, 3) for k, v in mk.items()}); rows.append(m)
    print(lab, m['net'], m['n'], m['pf'], f'{time.time()-t0:.1f}s', flush=True)
df = pd.DataFrame(rows).set_index('lab'); df.to_pickle(f'f1_{tf}.pkl')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.drop(columns=['yr']).to_string()); print(df['yr'].to_string())
