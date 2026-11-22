import numpy as np, pandas as pd, json
import v55
from v55 import evalp
from val55 import battery
from rg55 import vote2, AGREE, ADX60, X15, A
from v54 import CFG
from scan import comp
from vote import vote_pos
def sc_of(x11):
    p11 = dict(CFG['p'][11]); p11['x'] = x11
    X = np.vstack([np.stack([comp(j, p11 if j == 11 else CFG['p'][j]) for j in range(1, 15)]), X15[None, :]]); return vote_pos(X, np.array(CFG['w'] + [1.0], float), CFG['on'], CFG['off'], 1)[1]
SCs = {2.0: sc_of(2.0), 2.5: sc_of(2.5), 2.25: sc_of(2.25), 3.0: sc_of(3.0)}
def setT(x11, on, onT, aT, ag=AGREE, adx=ADX60): v55.TG = vote2(SCs[x11], adx, ag, 0.0, aT, on, on, onT, on, 0.0)
K = ['net', 'n', 'pf', 'win', 'sharpe', 'sortino', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'posM', 'last12', 'yr', 'b2min', 'b2std', 'b2neg', 'b6', 'b6min', 'stres', 'stresOOS', 'mae', 'monkey_p', 'mc_p5', 'mc_dd95']
for lab, cfg in [('V5.5', (2.0, 0.08, 0.08, 25.0)), ('Q', (2.5, 0.08, 0.12, 25.0)), ('R', (2.5, 0.04, 0.12, 25.0))]:
    setT(*cfg); o, R, b2, b6 = battery(A); print(lab, {k: o[k] for k in K}, flush=True)
for lab, base in [('Q', (2.5, 0.08, 0.12, 25.0)), ('R', (2.5, 0.04, 0.12, 25.0))]:
    rows = []
    for i, vals in enumerate([[2.25, 3.0], [base[1] - 0.01, base[1] + 0.01], [0.10, 0.15], [20.0, 30.0]]):
        for v in vals:
            c = list(base); c[i] = v; setT(*c); e = evalp(A); rows.append(dict(p=f'{["x11","on","onT","aT"][i]}={v}', net=e['net'], OOS=e['OOS'], L12=e['L12'], p98=e['p98'], mddc=e['mddc']))
    for k in (1, 2):
        setT(*base, ag=np.r_[np.zeros(k, np.int64), AGREE[:-k]], adx=np.r_[np.zeros(k), ADX60[:-k]]); e = evalp(A); rows.append(dict(p=f'gecikme {k}', net=e['net'], OOS=e['OOS'], L12=e['L12'], p98=e['p98'], mddc=e['mddc']))
    print(lab, 'plato/gecikme:'); print(pd.DataFrame(rows).to_string(index=False))
