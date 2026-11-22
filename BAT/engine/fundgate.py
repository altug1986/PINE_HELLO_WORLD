import numpy as np, pandas as pd
from fundhunt import D, c, t, run, F, Fz, side, Fr, apply25
from p1m import report
rows = [report(D, apply25(D, run(D, ovr=side), Fr), 'V7.1 (motor)')]
Z = np.zeros(len(c), bool)
V = [('Funding ≥ %0.03 iken yeni long yok', np.nan_to_num(F) >= 0.03, Z), ('Funding ≥ %0.05 iken yeni long yok', np.nan_to_num(F) >= 0.05, Z),
     ('Funding z ≤ −2 iken yeni short yok', Z, np.nan_to_num(Fz) <= -2), ('Funding z ≤ −1.5 iken yeni short yok', Z, np.nan_to_num(Fz) <= -1.5),
     ('İkisi (≥%0.03 / z ≤ −2)', np.nan_to_num(F) >= 0.03, np.nan_to_num(Fz) <= -2)]
for lab, gl, gs in V: rows.append(report(D, apply25(D, run(D, ovr=side, xGL=gl, xGS=gs), Fr), lab))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
