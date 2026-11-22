import numpy as np, pandas as pd, json
from f2 import *
from met import metrics, regimes, monkey, SPLIT
D = load(5); B = json.load(open('f2_best.json')); prm = B['prm']; prm['Z96'] = dict(n=96, k=3.5, ex=0.5, sl=8, mb=288)
def mot(m): return motor(D, 'Z', prm['Z96']) if m == 'Z96' else motor(D, m, prm[m])
RG = regimes(D); rows = []
for lab, order in [('Z96', ['Z96']), ('LİK>Z96', ['LIQ', 'Z96']), ('LİK>Z96>RSI', ['LIQ', 'Z96', 'RSI']), ('LİK>Z96>Z576', ['LIQ', 'Z96', 'Z'])]:
    R = simR(D, chain([mot(m) for m in order])); m = metrics(D, R, lab)
    xt = pd.to_datetime(D['t'][R['xb']], unit='ms'); p = pd.Series(R['pnl'], index=xt); b = p.groupby([xt.year, (xt.month - 1) // 6]).sum()
    q = R['q']; st = R['pnl'] - 0.0004 * q * (R['ep'] + R['xp']) - 4 * TICK * q
    mk = monkey(D, R, RG, 1000)
    rows.append(dict(sistem=lab, **{k: m[k] for k in ['net', 'n', 'pf', 'sharpe', 'sortino', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee']},
                     blok_poz=f'{(b>0).sum()}/{len(b)}', stres=round(st.sum()), monkey_max_p=round(max(mk.values()), 3), yr=m['yr']))
df = pd.DataFrame(rows).set_index('sistem'); pd.set_option('display.width', 250); print(df.T.to_string())
