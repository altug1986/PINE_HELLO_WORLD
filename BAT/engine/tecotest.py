import numpy as np, pandas as pd, itertools, v55
from donhold import D, c, n
from short_top_lib import to5, zh, DIVh, q0, simR, run
from teco import daily, mask_from
from eng import equity
from met import SPLIT
from v55 import t
a = np.nan_to_num(to5(zh(DIVh, 720))); b = np.nan_to_num(to5(zh(DIVh, 24))); WL = (a >= 1) | (b >= 1.5); WS = (a <= -1) | (b <= -1.5)
def runR(GL, GS): v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); return simR(D, run(q0)[0])
R0 = runR(WL, WS); M = mask_from(daily(R0), 7, 3)                                  # büyük kâr haftaları: V6.1'den, SABİT
def m30layers(R, lab):
    o = R['open']; cl, mtm, worst = equity(n, R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl'], D['h'], D['l'], D['c'], o[0], o[1], o[2], o[3])
    E = pd.Series(mtm, index=pd.to_datetime(t + 300000, unit='ms')).resample('30min').last().ffill(); d = E.diff().fillna(0)
    inm = M.reindex(d.index.normalize()).fillna(False).to_numpy(); yrs = (t[-1] - t[0]) / 3.15576e10; k = np.sqrt(48 * 365.25); out = dict(t=lab)
    for nm, dd in [('TAM', d), ('NORMAL', d.where(~inm, 0.0))]:
        eq = dd.cumsum(); uw = eq.cummax() - eq; net = dd.sum()
        out.update({f'{nm} net': round(net), f'{nm} MaxDD': round(uw.max()), f'{nm} Calmar': round(net / yrs / uw.max(), 2), f'{nm} Sharpe': round(dd.mean() / dd.std() * k, 2), f'{nm} P98': round(uw.quantile(.98))})
    out['BÜYÜK net'] = round(d[inm].sum()); out['son200 NORMAL'] = round(d.where(~inm, 0.0)[d.index >= d.index[-1] - pd.Timedelta(days=200)].sum()); return out
rows = [m30layers(R0, 'V6.1')]
ad = np.abs(np.diff(c, prepend=c[0])); cs = np.cumsum(ad)
for N, qq in itertools.product([288, 576, 864], [0.8, 0.9]):
    ret = np.r_[np.full(N, 0.0), (c[N:] / c[:-N] - 1) * 100]; er = np.nan_to_num(np.abs(np.r_[np.full(N, 0.0), c[N:] - c[:-N]]) / np.r_[np.full(N, np.nan), cs[N:] - cs[:-N]])
    e = np.quantile(er[(t < SPLIT) & (er > 0)], qq); dn = (er >= e) & (ret <= -3); up = (er >= e) & (ret >= 3)
    rows.append(m30layers(runR(WL | dn, WS | up), f'kararlı trende ters giriş yok: {N//288} gün, %{int(qq*100)}'))
r = pd.DataFrame(rows); r['BÜYÜK koruma %'] = (r['BÜYÜK net'] / r['BÜYÜK net'].iloc[0] * 100).round(0)
pd.set_option('display.width', 260); print('büyük kâr haftası:', int(M.sum()), 'gün'); print(r.to_string(index=False))
# 200 trende ters işlemin normal / büyük dağılımı
N = 576; ret = np.r_[np.full(N, 0.0), (c[N:] / c[:-N] - 1) * 100]; er = np.nan_to_num(np.abs(np.r_[np.full(N, 0.0), c[N:] - c[:-N]]) / np.r_[np.full(N, np.nan), cs[N:] - cs[:-N]])
e = np.quantile(er[(t < SPLIT) & (er > 0)], 0.9); i = R0['eb']; s = R0['dir']; p = R0['pnl']; ct = (er[i] >= e) & (ret[i] * s <= -3)
xd = pd.to_datetime(t[R0['xb']], unit='ms').normalize(); inm = M.reindex(xd).fillna(False).to_numpy()
print(f'trende ters 200 işlem: büyük haftalarda {(ct & inm).sum()} işlem {p[ct & inm].sum():.0f}$ | normal dönemde {(ct & ~inm).sum()} işlem {p[ct & ~inm].sum():.0f}$ (işlem başı {p[ct & ~inm].mean():.2f}$)')
print(f'diğer işlemler normal dönemde: {(~ct & ~inm).sum()} işlem {p[~ct & ~inm].sum():.0f}$ (işlem başı {p[~ct & ~inm].mean():.2f}$)')
