import itertools, numpy as np, pandas as pd
from edge2 import D, c, t, run, report, fast, ent, sig, A
b = fast(run(D), 'V6.6'); rows = []
for kind, aa in A.items():
    for a, H, sl in itertools.product(aa, (144, 288, 576), (5.0, 10.0)):
        rows.append(fast(run(D, ovr=ent(sig(kind, a), c, H, sl)), kind, a=str(a), H=H, sl=sl))
r = pd.DataFrame(rows); r.to_pickle('edge3.pkl'); pd.set_option('display.width', 260); print(b)
print(r.groupby('t').agg(ort=('net', 'mean'), max=('net', 'max'), baz_ustu=('net', lambda v: int((v > b['net']).sum())), y6=('net', lambda v: int((v > b['net'] * 1.06).sum())), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean')).round(0).to_string())
for kind in A: print(kind); print(r[r.t == kind].pivot_table(index=['a', 'H'], columns='sl', values='net').round(0).to_string())
