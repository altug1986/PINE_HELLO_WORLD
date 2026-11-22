import numpy as np, pandas as pd
from edge2 import D, c, t, run, trades, ent, sig
cap = ent(sig('Kapitülasyon', (5, 4)), c, 144, 5.0); btc = ent(sig('BTC çöküşü', 10), c, 288, 10.0); h = D['h']; l = D['l']; n = len(c); yrs = (t[-1] - t[0]) / 3.15576e10
rows = []
for lab, kw in [('V6.3', dict(useST=0)), ('V6.6', {}), ('V6.8', dict(ovr=cap)), ('V6.6 + BTC çöküşü', dict(ovr=btc)), ('V6.8 + BTC çöküşü', dict(ovr=np.where(cap != 0, cap, btc)))]:
    R, op = trades(D, run(D, **kw)); cum = np.zeros(n); np.add.at(cum, R['xb'], R['pnl']); cum = np.cumsum(cum); opn = np.zeros(n)
    for eb, xb, d, ep, q in zip(R['eb'], R['xb'], R['dir'], R['ep'], R['q']): s = slice(eb + 1, xb); opn[s] = (c[s] - ep) * q * d
    E = pd.Series(cum + opn, index=pd.to_datetime(t + 300000, unit='ms')).resample('30min').last().ffill(); uw = E.cummax() - E
    x = np.arange(len(E)); r2 = np.corrcoef(x, E.to_numpy())[0, 1] ** 2; mo = E.resample('ME').last().diff().dropna(); cv = mo.std() / mo.mean(); net = R['pnl'].sum()
    p98 = uw.quantile(.98); bob = (net / yrs / p98) * r2 ** 2 * (mo > 0).mean() / (1 + 0.25 * min(cv, 8))
    rows.append(dict(sürüm=lab, net=round(net), **{'yıllık net / DD P98': round(net / yrs / p98, 2), 'R²': round(r2, 3), 'pozitif ay %': round((mo > 0).mean() * 100), 'aylık CV': round(cv, 2), 'BoB': round(bob, 3), 'MaxDD': round(uw.max())}))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).sort_values('BoB', ascending=False).to_string(index=False))
