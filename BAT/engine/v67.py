import numpy as np, pandas as pd, v55
from tvcmp63 import tvx, Rtv, Mdf
from hatatrend import vote_x, SC, ADX60, AGREE, SH, STT, Z, q0, TS, simR, run, D
from stlib import GL0, GS0
from v55 import t
from mtm30 import m30
SPL = (pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
p = pd.read_excel('tv/v67_tv.xlsx', 'Performans').set_index('Unnamed: 0')['Tümü USDT']
print({k: p.get(k) for k in ['Net kar', 'Ödenen komisyon', 'Maks düşüş (kapanıştan kapanışa)', 'Maks düşüş (çubuk içi)']})
rows = []
for lab, f in [('V6.3 (TV)', 'tv/v63_tv.xlsx'), ('V6.7 TREND (TV)', 'tv/v67_tv.xlsx')]:
    T = tvx(f); pl = T.pnl; end = T.xt.max(); m = m30(Rtv(T)[0])
    rows.append({'sürüm': lab, 'net': round(pl.sum()), 'işlem': len(T), 'PF': round(pl[pl > 0].sum() / -pl[pl < 0].sum(), 3), 'IS/OOS': f"{pl[T.et < SPL].sum():.0f}/{pl[T.et >= SPL].sum():.0f}",
                 'son12ay': round(pl[T.xt >= end - 365 * 86400000].sum()), 'son200g': round(pl[T.xt >= end - 200 * 86400000].sum()), 'MaxDD30': m['MaxDD 30dk mum içi'], 'Calmar30': m['Calmar 30dk'],
                 'DD95/98/99': m['DD P95/P98/P99 30dk'], 'Sh/So': m['Sharpe/Sortino 30dk'], 'yıl': T.groupby(pd.to_datetime(T.xt, unit='ms').dt.year).pnl.sum().round(0).astype(int).to_dict()})
pd.set_option('display.width', 300); print(pd.DataFrame(rows).to_string(index=False))
# V6.6 çekirdeği (motor, TV benzeri balina) ile eşleşme → V6.7'nin eklediği / değiştirdiği işlemler
v55.TG = vote_x(SC, ADX60, AGREE, SH, Z, STT, Z, 20.0, 0.04, 0.15, 0.12, -0.15); v55.GL = GL0.astype(np.int64); v55.GS = GS0.astype(np.int64)
M = Mdf(simR(D, run(dict(q0, **TS))[0])); T = tvx('tv/v67_tv.xlsx'); T = T[T.xt <= t[-1]]
km = set(zip(M.et, M.dr)); T['m'] = [k in km for k in zip(T.et, T.dr)]
print(f'motor V6.6 (TV benzeri): {len(M)} işlem {M.pnl.sum():.0f}$ | TV V6.7 {len(T)} işlem, V6.6 girişiyle eşleşen %{T.m.mean()*100:.1f}')
x = T[~T.m]; print(f'V6.6\'da olmayan TV işlemleri: {len(x)} adet, {x.pnl.sum():.0f}$, ort {x.pnl.mean():.2f}$, kazanan %{(x.pnl>0).mean()*100:.0f}')
print(x.groupby(pd.to_datetime(x.et, unit='ms').dt.year).pnl.agg(['size', 'sum']).round(0).T.to_string())
