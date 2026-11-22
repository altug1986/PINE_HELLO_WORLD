import numpy as np, pandas as pd
from feat import *
from met import SPLIT
T = pd.read_pickle('tv/v54_tv.pkl'); T['i'] = np.searchsorted(t, T.et); print('motor veri sonu', pd.to_datetime(t[-1], unit='ms'), '| kapsam dışı TV işlemi', (T.i >= n).sum()); T = T[T.i < n].copy(); T['x'] = pd.to_datetime(T.xt, unit='ms')
for k, v in F.items(): T[k] = v[T.i]
for k in ['m864', 'm2016', 'di15', 'bdi', 'b288', 'b72', 'd288']: T[k + 'A'] = T[k] * T.dr     # pozisyon yönüne göre (+ = lehte)
T['blk'] = (T.x.dt.year * 12 + T.x.dt.month - 1) // 2; T['hr'] = pd.to_datetime(T.et, unit='ms').dt.hour; T['IS'] = T.xt < SPLIT
B = T.groupby('blk').pnl.agg(['sum', 'count']); B.index = [f"{b*2//12}-{b*2%12+1:02d}" for b in B.index]
print('2 AYLIK BLOK (TV):', len(B), 'blok | ort', round(B['sum'].mean()), '| std', round(B['sum'].std()), '| min', round(B['sum'].min()), '| negatif', (B['sum'] < 0).sum(), '| <50$:', (B['sum'] < 50).sum())
print(' '.join(f"{i}:{v:.0f}" for i, v in B['sum'].items()))
w = B['sum'].nsmallest(8); print('EN ZAYIF 8:', ' '.join(f"{i}:{v:.0f}" for i, v in w.items()))
T.to_pickle('tv/v54_feat.pkl')
def tab(col, q=5):
    x = T[col]; ok = x.notna(); b = pd.qcut(x[ok], q, duplicates='drop')
    g = T[ok].groupby(b, observed=True).agg(n=('pnl', 'size'), net=('pnl', 'sum'), avg=('pnl', 'mean'), IS=('pnl', lambda s: s[T.loc[s.index, 'IS']].sum()), OOS=('pnl', lambda s: s[~T.loc[s.index, 'IS']].sum()))
    return g.round(1)
pd.set_option('display.width', 200)
for col in ['bars', 'mae', 'adx15', 'di15A', 'badx', 'bdiA', 'nr', 'volr', 'm864A', 'm2016A', 'b288A', 'd288A']:
    print('\n#', col); print(tab(col).to_string())
h = T.groupby('hr').pnl.agg(['sum', 'count']); print('\n# saat (UTC) net:', ' '.join(f"{i}:{v:.0f}" for i, v in h['sum'].items()))
