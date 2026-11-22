import numpy as np, pandas as pd, sys
from v54 import run, D, metrics, simR, SPLIT, TICK
P54 = dict(L=5.0, a=40.0, vm=1, vk=2.0, cS=5.0, N=288, w15=1.0, k15=1.5, a15=4.0)
def tvload(f):
    x = pd.read_csv(f); x['ts'] = pd.to_datetime(x['Tarih ve saat'], errors='coerce'); op = x[x.ts.isna()]['İşlem numarası']; x = x[~x['İşlem numarası'].isin(op)]
    x['ms'] = ((x.ts - pd.Timedelta(hours=3) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1ms')).astype(np.int64)
    e = x[x.Tip.str.contains('giriş')].set_index('İşlem numarası'); q = x[x.Tip.str.contains('çıkış')].set_index('İşlem numarası')
    T = pd.DataFrame(dict(et=e.ms, xt=q.ms, dr=np.where(e.Tip.str.contains('Kısa'), -1, 1), ep=e['Fiyat USDT'], xp=q['Fiyat USDT'], pnl=q['Net K/Z USDT'], mae=q['Tersine hareket %'], bars=q['Süre (çubuk)']))
    return T.sort_values('et').reset_index()
if __name__ == '__main__':
    T = tvload('tv/v54_tv.csv'); t = D['t']
    cp = np.r_[0, T.pnl.cumsum()]; print('TV:', len(T), 'işlem | net', round(T.pnl.sum()), '| PF', round(T.pnl[T.pnl > 0].sum() / -T.pnl[T.pnl < 0].sum(), 3), '| MaxDD kapanış', round((np.maximum.accumulate(cp) - cp).max()),
          '| ilk', pd.to_datetime(T.et.iloc[0], unit='ms'), 'son', pd.to_datetime(T.xt.iloc[-1], unit='ms'))
    tg, nr = run(P54); R = simR(D, tg)
    M = pd.DataFrame(dict(et=t[R['eb']], xt=t[R['xb']], dr=R['dir'], pnl=R['pnl']))
    a, b = T.et.min(), T.xt.max(); Mm = M[(M.et >= a) & (M.xt <= b)]
    print('Motor aynı dönem:', len(Mm), 'işlem | net', round(Mm.pnl.sum()))
    k = set(zip(T.et, T.dr)); km = set(zip(Mm.et, Mm.dr)); both = k & km
    print('giriş eşleşmesi', len(both), f'| TV kapsama %{len(both)/len(T)*100:.1f} | motor kapsama %{len(both)/len(Mm)*100:.1f}')
    j = T.merge(Mm, on=['et', 'dr'], suffixes=('_tv', '_m'))
    same = j[j.xt_tv == j.xt_m]; dif = j[j.xt_tv != j.xt_m]
    print('aynı giriş+çıkış', len(same), 'Δpnl', round((same.pnl_tv - same.pnl_m).sum(), 1), '| aynı giriş farklı çıkış', len(dif), 'Δpnl', round((dif.pnl_tv - dif.pnl_m).sum(), 1))
    oT = T[~T.set_index(['et', 'dr']).index.isin(list(both))]; oM = Mm[~Mm.set_index(['et', 'dr']).index.isin(list(both))]
    print('sadece TV', len(oT), round(oT.pnl.sum()), '| sadece motor', len(oM), round(oM.pnl.sum()))
    T.to_pickle('tv/v54_tv.pkl')
