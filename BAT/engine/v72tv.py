import numpy as np, pandas as pd
from tvcmp63 import tvx, Rtv
from mtm30 import m30
from v55 import t
SPL = (pd.Timestamp('2024-11-01') - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms'); rows = []; B = {}
for lab, f in [('V7.1', 'tv/v71_tv.xlsx'), ('V7.2', 'tv/v72_tv.xlsx')]:
    T = tvx(f); p = pd.read_excel(f, 'Performans').set_index('Unnamed: 0')['Tümü USDT']; pl = T.pnl; end = T.xt.max(); m = m30(Rtv(T)[0])
    x = pd.read_excel(f, 'İşlemler'); n = x[x['Tarih ve saat'].astype(str) == 'Açık']['İşlem numarası']; op = x[x['İşlem numarası'].isin(n) & x.Tip.str.contains('giriş')]
    mo = pd.Series(pl.values, index=pd.to_datetime(T.xt, unit='ms')).resample('ME').sum()
    B[lab] = pd.Series(pl.values, index=pd.to_datetime(T.xt, unit='ms')).resample('6MS').sum().round(0).astype(int)
    rows.append({'sürüm': lab, 'net (TV)': round(p['Net kar']), 'işlem': len(T), 'PF': round(pl[pl > 0].sum() / -pl[pl < 0].sum(), 3), 'MaxDD TV kapanış/mum içi': f"{p['Maks düşüş (kapanıştan kapanışa)']:.0f}/{p['Maks düşüş (çubuk içi)']:.0f}",
                 'IS/OOS': f"{pl[T.et < SPL].sum():.0f}/{pl[T.et >= SPL].sum():.0f}", 'son 30 g': round(pl[T.xt >= end - 30 * 86400000].sum()), 'son 200 g': round(pl[T.xt >= end - 200 * 86400000].sum()), 'son 365 g': round(pl[T.xt >= end - 365 * 86400000].sum()),
                 'pozitif ay %': round((mo > 0).mean() * 100), 'aylık CV': round(mo.std() / mo.mean(), 2), 'MaxDD30': m['MaxDD 30dk mum içi'], 'Calmar30': m['Calmar 30dk'], 'DD95/98/99': m['DD P95/P98/P99 30dk'], 'Sh/So': m['Sharpe/Sortino 30dk'],
                 'açık poz': f"{'L' if 'Uzun' in op.Tip.iloc[0] else 'S'} {pd.Timestamp(op['Tarih ve saat'].iloc[0]):%d.%m %H:%M} @{op['Fiyat USDT'].iloc[0]} → {p['Açık K/Z']:+.0f}$"})
pd.set_option('display.width', 300); print(pd.DataFrame(rows).set_index('sürüm').T.to_string())
b = pd.DataFrame(B); b.index = b.index.strftime('%Y-%m'); print('\n6 aylık bloklar $:'); print(b.T.to_string())
print('V7.2 − V7.1 blok farkı:', (b['V7.2'] - b['V7.1']).to_dict())
