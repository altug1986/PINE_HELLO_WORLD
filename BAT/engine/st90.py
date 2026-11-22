import itertools, numpy as np, pandas as pd
from stlib import ev, st5, D
A30 = np.nan_to_num(D['adx30'])
M = [('tut', dict(offW=-0.15)), ('ters .08', dict(onCS=0.08)), ('ters .12', dict(onCS=0.12)), ('ters .15', dict(onCS=0.15)), ('kapı', dict(filt=True))]
rows = [ev(None, lab='baz')]
for f, a in itertools.product((2.0, 2.5, 3.0, 3.5, 4.0), (0, 20, 25, 30, 35)):
    s = st5(1.5, 10, f) * (A30 >= a)
    for lab, kw in M: rows.append(dict(ev(s, lab=lab, **kw), fac=f, adx=a))
r = pd.DataFrame(rows); r.to_pickle('st90.pkl'); b = r.iloc[0]; pd.set_option('display.width', 250)
print(r.iloc[:1].to_string(index=False))
print(r.iloc[1:].pivot_table(index=['t', 'adx'], columns='fac', values='net').round(0).to_string())
print(r.iloc[1:].sort_values('IS', ascending=False).head(10)[['t', 'fac', 'adx', 'net', 'n', 'IS', 'OOS', 'L12', 'L200', 'DD', 'Cal', 'P', 'SS']].to_string(index=False))
