import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B); c = F['c']
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
eq, t = z01.run2(B, F, 20, 10, 4, 99, 120)
i = t.ei.astype(int).values
X = pd.DataFrame(index=t.index)
X['trend2g'] = (c / pd.Series(c).ewm(span=1440, adjust=False).mean().values - 1)[i] * 100
X['trend7g'] = (c / pd.Series(c).ewm(span=5040, adjust=False).mean().values - 1)[i] * 100
oi = B.oi_zec.values; X['dOI_spike'] = (oi[i] / oi[np.maximum(i - 20, 0)] - 1) * 100
X['basis'] = B.basis_bps.values[i]
tp = B.tt_pos_ls.values; X['dTTpos'] = (np.log(tp[i]) - np.log(tp[np.maximum(i - 20, 0)])) * 100
td = pd.Series(B.taker_delta_base.values).rolling(20).sum().values / pd.Series(B.volume.values).rolling(20).sum().values
X['takerImb'] = td[i]
X['btc_mv'] = (B.btc_close.values[i] / B.btc_close.values[np.maximum(i - 20, 0)] - 1) * 100
X['bookImb'] = ((B.book_m1 - B.book_p1) / (B.book_m1 + B.book_p1)).values[i]
X['pnl'] = t.pnl.values; X['side'] = t.side.values; X['IS'] = (t.et < mid).values
for side, lab in ((-1, 'SHORT (pump fade)'), (1, 'LONG (dump fade)')):
    d = X[X.side == side]; print(f'== {lab}: n {len(d)} net {d.pnl.sum():.0f} | IS {d[d.IS].pnl.sum():.0f} OOS {d[~d.IS].pnl.sum():.0f}')
    for f in ['trend2g', 'trend7g', 'dOI_spike', 'basis', 'dTTpos', 'takerImb', 'btc_mv', 'bookImb']:
        dd = d.dropna(subset=[f])
        if len(dd) < 60: continue
        q = pd.qcut(dd[f], 3, labels=['düşük', 'orta', 'yüksek'], duplicates='drop')
        g = dd.groupby([q, 'IS'], observed=True).pnl.sum().unstack().reindex(columns=[True, False])
        print(f'  {f:10s} ' + ' | '.join(f'{k}: IS {g.loc[k, True]:6.0f} OOS {g.loc[k, False]:6.0f}' for k in g.index) + f'  (eşikler {np.round(dd[f].quantile([1/3, 2/3]).values, 2)})')
