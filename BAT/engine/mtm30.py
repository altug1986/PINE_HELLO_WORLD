import numpy as np, pandas as pd
import v55
from v55 import run, simR, D, t, n, evalp
from eng import equity
from met import SPLIT
def m30(R):
    o = R['open']; cl, mtm, worst = equity(n, R['eb'], R['xb'], R['dir'], R['ep'], R['q'], R['pnl'], D['h'], D['l'], D['c'], o[0], o[1], o[2], o[3])
    ix = pd.to_datetime(t + 300000, unit='ms'); E = pd.Series(mtm, index=ix).resample('30min', label='right', closed='right').last().ffill(); W = pd.Series(worst, index=ix).resample('30min', label='right', closed='right').min().reindex(E.index).fillna(E)
    pk = E.cummax(); uw = pk - E; uwW = pk - W; yrs = (t[-1] - t[0]) / 3.15576e10; net = E.iloc[-1]; d = E.diff().fillna(0); k = np.sqrt(48 * 365.25)
    sub = (uw > 0).astype(int); grp = (sub.diff() != 0).cumsum(); longest = (sub.groupby(grp).sum().max()) / 48
    clp = np.r_[0, np.cumsum(R['pnl'])]; mddc = (np.maximum.accumulate(clp) - clp).max()
    return {'net': round(net), 'MaxDD kapanmış': round(mddc), 'Calmar kapanmış': round(net / yrs / mddc, 2), 'MaxDD 30dk': round(uw.max()), 'MaxDD 30dk mum içi': round(uwW.max()),
            'Calmar 30dk': round(net / yrs / uw.max(), 2), 'DD P95/P98/P99 30dk': f"{uw.quantile(.95):.0f}/{uw.quantile(.98):.0f}/{uw.quantile(.99):.0f}",
            'Sharpe/Sortino 30dk': f"{d.mean() / d.std() * k:.2f}/{d.mean() / d[d < 0].std() * k:.2f}", 'Ulcer/UPI 30dk': f"{np.sqrt((uw ** 2).mean()):.0f}/{net / yrs / np.sqrt((uw ** 2).mean()):.1f}",
            'zirve altı zaman %': round(sub.mean() * 100), 'en uzun zirve altı (gün)': round(longest)}
if __name__ == '__main__':
    from v54 import target
    from base56b import A, SCQ
    from val56 import SCs
    from rg55 import vote2, AGREE, ADX60
    V = []
    v55.TG = target(1.0, 1.5, 4.0); V.append(('V5.4 (TV 6921)', {}))
    V.append(('V5.5 (TV 7448)', A))
    rows = []
    for lab, p in V: tg, _ = run(p); rows.append(dict(sürüm=lab, **m30(simR(D, tg))))
    for lab, on, p in [('V5.6 (motor)', 0.04, A), ('V5.6B (TV 8185)', 0.08, A), ('V5.6B + stop %6', 0.08, dict(A, S=6.0))]:
        v55.TG = vote2(SCs[2.5], ADX60, AGREE, 0.0, 25.0, on, on, 0.12, on, 0.0); tg, _ = run(p); rows.append(dict(sürüm=lab, **m30(simR(D, tg))))
    pd.set_option('display.width', 300); print(pd.DataFrame(rows).set_index('sürüm').T.to_string())
