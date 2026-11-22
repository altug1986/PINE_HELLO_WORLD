import numpy as np, pandas as pd
from monthly import Zs
from crossscan import D, c, t, Fr, ent, deep, v71
from p1m import report, trades
def sig(k, lag=0):
    s = ((Zs['bo48'] <= -k) & (Zs['bp48'] <= -k)) | ((Zs['ao48'] <= -k) & (Zs['ap48'] <= -k)); s = np.where(deep == -1, 0, np.nan_to_num(s)).astype(np.int64)
    return np.r_[np.zeros(lag, np.int64), s[:-lag]] if lag else s
rows = [report(D, v71(), 'V7.1')]; best = ent(sig(2.5), c, 288, 5.0); ex = v71(new=best); rows.append(report(D, ex, 'V7.2: BAT|BTC z≤−2.5 · 24 s · stop %5'))
for lab, e in [('  z −2.25', ent(sig(2.25), c, 288, 5.0)), ('  z −2.75', ent(sig(2.75), c, 288, 5.0)), ('  tutma 18 s', ent(sig(2.5), c, 216, 5.0)), ('  tutma 30 s', ent(sig(2.5), c, 360, 5.0)),
               ('  stop %4', ent(sig(2.5), c, 288, 4.0)), ('  stop %7', ent(sig(2.5), c, 288, 7.0)), ('  veri 1 mum gecikmeli', ent(sig(2.5, 1), c, 288, 5.0)), ('  veri 2 mum gecikmeli', ent(sig(2.5, 2), c, 288, 5.0))]:
    rows.append(report(D, v71(new=e), lab))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
R0, _ = trades(D, v71()); R1, _ = trades(D, ex)
y = lambda R: pd.Series(R['pnl'], index=pd.to_datetime(t[R['xb']], unit='ms').year).groupby(level=0).sum()
print('yıl Δ$:', (y(R1) - y(R0)).round(0).astype(int).to_dict())
on = np.flatnonzero((best != 0) & (np.r_[0, best[:-1]] == 0)); mo = pd.Series(1, index=pd.to_datetime(t[on], unit='ms').to_period('M')).groupby(level=0).size()
allm = pd.period_range(mo.index.min(), pd.Timestamp(t[-1], unit='ms').to_period('M'), freq='M'); mo = mo.reindex(allm, fill_value=0)
print(f'yan motor devreye girişi: toplam {len(on)} | ayda ort. {mo.mean():.2f} | girişli ay %{(mo > 0).mean()*100:.0f} | son 12 ay: {mo.iloc[-12:].tolist()}')
