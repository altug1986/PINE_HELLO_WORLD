import numpy as np, pandas as pd
from astre_test import astre, TS, R_
from v55 import t
GL, GS = astre(-300000); s1 = t >= (pd.Timestamp('2026-04-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
for lab, gl, gs in [('TV benzeri (V6.3 TV)', GL & s1, GS & s1), ('tam kapı (V6.4)', GL, GS)]:
    _, R = R_(gl, gs, TS); xt = t[R['xb']]; p = pd.Series(R['pnl']); e = t[-1]
    y = p.groupby(pd.to_datetime(xt, unit='ms').year).sum().round(0).astype(int).to_dict()
    print(lab, '| net', round(p.sum()), '| son12ay', round(p[xt >= e - 365 * 86400000].sum()), '| son200g', round(p[xt >= e - 200 * 86400000].sum()), '|', y)
