import numpy as np, pandas as pd
from edge2 import D, c, t, run, report, trades, ent, sig
from p1m import MS
cap = ent(sig('Kapitülasyon', (5, 4)), c, 144, 5.0); h = D['h']; l = D['l']
rows = []
for lab, kw in [('V6.3', dict(useST=0)), ('V6.6', {}), ('V6.8', dict(ovr=cap))]:
    out = run(D, **kw); o = report(D, out, lab); R, op = trades(D, out); n = len(c)
    cum = np.zeros(n); np.add.at(cum, R['xb'], R['pnl']); cum = np.cumsum(cum); opn = np.zeros(n); mae = []
    for eb, xb, d, ep, q in zip(R['eb'], R['xb'], R['dir'], R['ep'], R['q']):
        s = slice(eb + 1, xb); opn[s] = (c[s] - ep) * q * d
        if xb > eb + 1: mae.append((((l[s] if d > 0 else h[s]) / ep - 1) * d * 100).min())
    E = pd.Series(cum + opn, index=pd.to_datetime(t + 300000, unit='ms')).resample('30min').last().ffill(); uw = E.cummax() - E; sub = uw > 0
    grp = (sub != sub.shift()).cumsum(); longest = sub.groupby(grp).sum().max() / 48
    mo = E.resample('ME').last().diff().dropna(); wk = E.resample('W').last().diff().dropna(); mae = np.array(mae)
    o.update({'zirve altı zaman %': round(sub.mean() * 100), 'en uzun zirve altı (gün)': round(longest), 'pozitif ay %': round((mo > 0).mean() * 100), 'pozitif hafta %': round((wk > 0).mean() * 100),
              'en kötü ay $': round(mo.min()), 'işlem içi en kötü anlık %': round(mae.min(), 1), 'anlık ≤ −%10 işlem': int((mae <= -10).sum())})
    rows.append(o)
pd.set_option('display.width', 300); print(pd.DataFrame(rows).T.to_string())
