import itertools, pandas as pd
from stlib import ev, st5
rows = [ev(None, lab='baz')]; print('nötr:', ev(st5(1, 10, 3.0), lab='nötr')['net'])
M = [('kapı', dict(filt=True))] + [(f'ters eşik {x}', dict(onCS=x)) for x in (0.08, 0.12, 0.15, 0.20, 0.30)] + [(f'yön eşik {x}', dict(onW=x)) for x in (0.02, 0.03)] + \
    [(f'tut {x}', dict(offW=x)) for x in (-0.04, -0.08, -0.15, -0.25)] + [('yön.03+tut-.08', dict(onW=0.03, offW=-0.08)), ('ters.12+tut-.08', dict(onCS=0.12, offW=-0.08)),
    ('ters.12+yön.03', dict(onCS=0.12, onW=0.03)), ('ters.12+yön.03+tut-.08', dict(onCS=0.12, onW=0.03, offW=-0.08))]
for h, f in itertools.product((1, 2, 3), (2.0, 3.0, 4.0)):
    s = st5(h, 10, f)
    for lab, kw in M: rows.append(dict(ev(s, lab=lab, **kw), tf=h, fac=f))
r = pd.DataFrame(rows); r.to_pickle('stscan.pkl'); b = r.iloc[0]
pd.set_option('display.width', 250); print(r.iloc[:1].to_string(index=False))
r['mod'] = r.t; g = r.iloc[1:].groupby('mod').agg(net_ort=('net', 'mean'), net_max=('net', 'max'), IS_ort=('IS', 'mean'), OOS_ort=('OOS', 'mean'), L12_ort=('L12', 'mean'), DD_ort=('DD', 'mean'), baz_ustu=('net', lambda x: int((x > b.net).sum())))
print(g.round(0).sort_values('net_ort', ascending=False).to_string())
print(r.iloc[1:].sort_values('IS', ascending=False).head(12)[['t', 'tf', 'fac', 'net', 'n', 'IS', 'OOS', 'L12', 'L200', 'DD', 'Cal', 'P', 'SS']].to_string(index=False))
