import itertools, numpy as np, pandas as pd
import btcliq as B
from fundhunt import D, c, t, oi, fw, H, SPL, pos
zs = lambda x: x / pd.Series(x).rolling(8640, min_periods=2000).std().to_numpy()          # 30 günlük oynaklığa göre
ao48, ap48, ao6, ap6 = B.ch(oi, 48), B.ch(c, 48), B.ch(oi, 6), B.ch(c, 6)
Zs = {k: zs(v) for k, v in dict(ao48=ao48, ap48=ap48, ao6=ao6, ap6=ap6, bo48=B.bo48, bp48=B.bp48, bo6=B.bo6, bp6=B.bp6).items()}
E = {}
for k in (2.0, 2.5, 3.0):
    E[f'BAT kapitülasyon z≤−{k}'] = (Zs['ao48'] <= -k) & (Zs['ap48'] <= -k)
    E[f'BTC kapitülasyon z≤−{k}'] = (Zs['bo48'] <= -k) & (Zs['bp48'] <= -k)
    E[f'BAT likidasyon 30dk z≤−{k}'] = (Zs['ao6'] <= -k) & (Zs['ap6'] <= -k)
    E[f'BTC likidasyon 30dk z≤−{k}'] = (Zs['bo6'] <= -k) & (Zs['bp6'] <= -k)
    E[f'BAT veya BTC kapitülasyon z≤−{k}'] = E[f'BAT kapitülasyon z≤−{k}'] | E[f'BTC kapitülasyon z≤−{k}']
    E[f'BAT short sıkışması z (OI≤−{k}, fiyat≥+{k}) [ters: short]'] = (Zs['ao48'] <= -k) & (Zs['ap48'] >= k)
valid = ~np.isnan(fw['24s']); rng = np.random.default_rng(9); pool = np.flatnonzero(valid & ~np.isnan(B.boi) & ~np.isnan(oi)); rows = []
yr = pd.to_datetime(t, unit='ms').year; mo = pd.to_datetime(t, unit='ms').to_period('M')
allm = pd.period_range(pd.Timestamp(t[pool[0]], unit='ms').to_period('M'), pd.Timestamp(t[pool[-1]], unit='ms').to_period('M'), freq='M')
for lab, cond in E.items():
    cond = np.nan_to_num(cond).astype(bool) & valid; ev = []; last = -10**9
    for i in np.flatnonzero(cond & ~np.r_[False, cond[:-1]]):
        if i - last >= 288: ev.append(i); last = i
    ev = np.array(ev, int)
    if len(ev) < 30: rows.append(dict(olay=lab, n=len(ev))); continue
    f = fw['24s'][ev]; f4 = fw['4s'][ev]; null = np.array([fw['24s'][rng.choice(pool, len(ev))].mean() for _ in range(2000)]); p = (np.sum(np.abs(null - null.mean()) >= abs(f.mean() - null.mean())) + 1) / 2001
    py = pd.Series(f, index=yr[ev]).groupby(level=0).agg(['size', 'mean']); em = pd.Series(1, index=mo[ev]).groupby(level=0).size().reindex(allm, fill_value=0)
    rows.append(dict(olay=lab, n=len(ev), **{'ay başına': round(len(ev) / len(allm), 2), 'olaylı ay %': round((em > 0).mean() * 100), '4s %': round(f4.mean(), 2), '24s %': round(f.mean(), 2), 'p': round(p, 3),
                     'pozitif yıl': f"{int((py['mean'] > 0).sum())}/{len(py)}", 'yıl 24s %': ' '.join(f"{y % 100}:{m_:+.1f}({s})" for y, (s, m_) in py.iterrows()), 'V7.1 aynı yön %': round((pos[ev] == np.sign(f.mean())).mean() * 100)}))
pd.set_option('display.width', 320); pd.set_option('display.max_colwidth', 80); print(pd.DataFrame(rows).to_string(index=False))
