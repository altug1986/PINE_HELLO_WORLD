import numpy as np, pandas as pd
from tvcmp63 import tvx, Rtv
from v55 import t
from mtm30 import m30
SPL = (pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')
rows = []
for lab, f in [('Astre V6.2 (TV)', 'tv/v62astre_tv.xlsx'), ('V6.3 (TV)', 'tv/v63_tv.xlsx')]:
    T = tvx(f); p = T.pnl; end = T.xt.max(); m = m30(Rtv(T)[0])
    y = T.groupby(pd.to_datetime(T.xt, unit='ms').dt.year).pnl.sum().round(0).astype(int).to_dict()
    rows.append({'sürüm': lab, 'net': round(p.sum()), 'işlem': len(T), 'PF': round(p[p > 0].sum() / -p[p < 0].sum(), 3), 'win %': round((p > 0).mean() * 100, 1),
                 'IS/OOS': f"{p[T.et < SPL].sum():.0f}/{p[T.et >= SPL].sum():.0f}", 'son12ay': round(p[T.xt >= end - 365 * 86400000].sum()), 'son200g': round(p[T.xt >= end - 200 * 86400000].sum()),
                 'MaxDD30 içi': m['MaxDD 30dk mum içi'], 'Calmar30': m['Calmar 30dk'], 'DD95/98/99': m['DD P95/P98/P99 30dk'], 'Sh/So': m['Sharpe/Sortino 30dk'], 'yıl': y})
pd.set_option('display.width', 300); print(pd.DataFrame(rows).to_string(index=False))
