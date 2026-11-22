import itertools, numpy as np, pandas as pd
from stlib import ev2, st5, D
A30 = np.nan_to_num(D['adx30']); rows = []
for f, aH, aT, x in itertools.product((2.5, 3.0, 3.5), (20, 25), (25, 30), (0.08, 0.12, 0.15)):
    s = st5(1.5, 10, f); rows.append(dict(ev2(s * (A30 >= aH), s * (A30 >= aT), x, lab='tut+ters'), fac=f, aH=aH, aT=aT, x=x))
r = pd.DataFrame(rows); r.to_pickle('st90b.pkl'); pd.set_option('display.width', 250)
print(r.pivot_table(index=['aH', 'aT', 'x'], columns='fac', values='net').round(0).to_string())
print(r.sort_values('IS', ascending=False).head(8)[['fac', 'aH', 'aT', 'x', 'net', 'n', 'IS', 'OOS', 'L12', 'L200', 'DD', 'Cal', 'P', 'SS']].to_string(index=False))
