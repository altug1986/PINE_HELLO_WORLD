import numpy as np, pandas as pd
import btcliq as B
from monthly3 import sig, D, c, t, Fr, ent, deep
from crossscan import run, apply25, capB, crash
from p1m import report, trades
capZ = ent(sig(2.5), c, 288, 5.0); flush = ent(np.where(deep == -1, 0, np.nan_to_num(B.bo288) <= -10).astype(np.int64), c, 288, 3.0)
def go(order):
    ov = crash.copy()
    for a in reversed(order): ov = np.where(a != 0, a, ov)
    return apply25(D, run(D, ovr=ov), Fr)
rows = [report(D, go([capB]), 'V7.1'), report(D, go([capB, flush]), 'V7.1 + BTC OI boşalması'), report(D, go([capB, capZ]), 'V7.1 + aylık kapitülasyon (z)'),
        report(D, go([capB, flush, capZ]), 'V7.2: + ikisi (öncelik boşalma > aylık)'), report(D, go([capB, capZ, flush]), 'V7.2b: + ikisi (öncelik aylık > boşalma)')]
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
R0, _ = trades(D, go([capB])); R1, _ = trades(D, go([capB, flush, capZ]))
y = lambda R: pd.Series(R['pnl'], index=pd.to_datetime(t[R['xb']], unit='ms').year).groupby(level=0).sum(); print('yıl Δ$ (V7.1 → V7.2):', (y(R1) - y(R0)).round(0).astype(int).to_dict())
