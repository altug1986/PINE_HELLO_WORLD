import itertools, numpy as np, pandas as pd, v55
import whsig as W
from whsig import F, ent, c, SC, DEN
def ev_add(p16, w, lab):                                      # seyreltmesiz: diğer bileşenlerin ağırlığı değişmez
    W.SC = SC + w / DEN * p16; o = W.ev(None, 0.0, lab); W.SC = SC; return o
import hold2
rows = [W.ev(lab='V6.6'), W.ev(np.zeros(len(SC)), 1.0, 'sadece seyreltme w=1')]
for (fn, x), k, yon, hold, w in itertools.product(F.items(), (2.0, 2.5, 3.0), ('ters', 'aynı'), (48, 144, 288), (1.0, 2.0, 4.0)):
    sg = np.where(x >= k, 1, np.where(x <= -k, -1, 0)) * (-1 if yon == 'ters' else 1)
    rows.append(dict(ev_add(ent(sg.astype(np.int64), c, hold, 10.0), w, f'{fn} {yon}'), k=k, hold=hold, w=w))
r = pd.DataFrame(rows); b = r.iloc[0]; pd.set_option('display.width', 250); print(r.iloc[:2].to_string(index=False))
g = r.iloc[2:].groupby('t').agg(ort=('net', 'mean'), max=('net', 'max'), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean'), DD_ort=('DD', 'mean'), baz_ustu=('net', lambda v: int((v > b.net).sum())))
print(g.round(0).to_string())
print(r.iloc[2:].sort_values('IS', ascending=False).head(8)[['t', 'k', 'hold', 'w', 'net', 'n', 'IS', 'OOS', 'L12', 'L200', 'DD', 'Cal', 'P', 'SS']].to_string(index=False))
r.to_pickle('whsig2.pkl')
