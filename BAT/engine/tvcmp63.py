import numpy as np, pandas as pd, v55
from astre_test import astre, WL, WS, q0, TS, R_
from v55 import t, n, D
from mtm30 import m30
def tvx(f):
    x = pd.read_excel(f, 'İşlemler'); x['ts'] = pd.to_datetime(x['Tarih ve saat'], errors='coerce'); op = x[x.ts.isna()]['İşlem numarası']; x = x[~x['İşlem numarası'].isin(op)]
    x['ms'] = ((x.ts - pd.Timedelta(hours=3) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).astype(np.int64)
    e = x[x.Tip.str.contains('giriş')].set_index('İşlem numarası'); q = x[x.Tip.str.contains('çıkış')].set_index('İşlem numarası')
    T = pd.DataFrame(dict(et=e.ms, xt=q.ms, dr=np.where(e.Tip.str.contains('Kısa'), -1, 1), ep=e['Fiyat USDT'].astype(float), q=e['Boyut (miktar)'].astype(float), pnl=q['Net K/Z USDT'].astype(float)))
    return T.sort_values('et').reset_index(drop=True)
def Rtv(T):
    T = T[T.xt <= t[-1]]; eb = np.searchsorted(t, T.et.values); xb = np.searchsorted(t, T.xt.values)
    return dict(eb=eb, xb=xb, dir=T.dr.values, ep=T.ep.values, q=T.q.values, pnl=T.pnl.values, open=(0, 0, 0.0, 0.0)), int((t[eb] != T.et.values).sum())
def Mdf(R): return pd.DataFrame(dict(et=t[R['eb']], xt=t[R['xb']], dr=R['dir'], pnl=R['pnl']))
GLb, GSb = astre(-300000); Z = np.zeros(n, bool)
if __name__ == '__main__':
    out = {}
    for lab, f, p in [('Astre V6.2', 'tv/v62astre_tv.xlsx', {}), ('V6.3', 'tv/v63_tv.xlsx', TS)]:
        T = tvx(f); Rt, bad = Rtv(T); Tm = T[T.xt <= t[-1]]
        _, Rg = R_(GLb, GSb, p); _, R0 = R_(Z, Z, p); Mg, M0 = Mdf(Rg), Mdf(R0)
        kt = set(zip(Tm.et, Tm.dr)); kg = set(zip(Mg.et, Mg.dr)); k0 = set(zip(M0.et, M0.dr))
        print(f'\n== {lab} | TV kapanmış {len(T)} net {T.pnl.sum():.0f} | motor sonuna kadar TV {len(Tm)} net {Tm.pnl.sum():.0f} | eşleşmeyen bar {bad}')
        print(f'motor kapılı {len(Mg)} net {Mg.pnl.sum():.0f} eşleşme %{len(kt & kg)/len(kt)*100:.1f} | motor kapısız {len(M0)} net {M0.pnl.sum():.0f} eşleşme %{len(kt & k0)/len(kt)*100:.1f}')
        blk = k0 - kg                                                    # motorda balina kapısının kestiği girişler
        b = pd.DataFrame(list(blk), columns=['et', 'dr']); b['ay'] = pd.to_datetime(b.et, unit='ms').dt.to_period('Q'); b['tv'] = [(e, d) in kt for e, d in zip(b.et, b.dr)]
        print('kapının kestiği girişler TV\'de var mı (çeyrek: kesilen / TV\'de olan):'); print(b.groupby('ay').tv.agg(['size', 'sum']).T.to_string())
        m = m30(Rt); out[lab] = m; print({k: m[k] for k in ['net', 'MaxDD 30dk', 'MaxDD 30dk mum içi', 'Calmar 30dk', 'DD P95/P98/P99 30dk', 'Sharpe/Sortino 30dk', 'Ulcer/UPI 30dk']})
