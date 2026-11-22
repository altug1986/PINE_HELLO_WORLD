import numpy as np, pandas as pd, numba as nb
import base56b, v55
from base56b import A
from v55 import run, simR, D, t
from met import SPLIT
from eng import TICK, COMM
m = pd.read_csv('BATUSDT_1DK_OHLCV_UTC.csv', usecols=['open_time_ms', 'open', 'high', 'low']); MT = m.open_time_ms.to_numpy(np.int64); MO, MH, ML = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy()
@nb.njit(cache=True)
def wt(s0, s1, dr, ep, xp, q, pnl, MO, MH, ML, sl, tick, comm):
    k = len(s0); out = pnl.copy(); mae = np.zeros(k); hit = np.zeros(k, np.int64)
    for j in range(k):
        d = dr[j]; e = ep[j]; lvl = e * (1 - sl / 100) if d > 0 else e * (1 + sl / 100); worst = e
        for i in range(s0[j], s1[j]):
            if d > 0:
                if sl > 0 and ML[i] <= lvl:
                    x = min(MO[i], lvl) - tick; worst = min(worst, x); out[j] = (x - e) * q[j] - comm * q[j] * (e + x); hit[j] = 1; break
                worst = min(worst, ML[i])
            else:
                if sl > 0 and MH[i] >= lvl:
                    x = max(MO[i], lvl) + tick; worst = max(worst, x); out[j] = (e - x) * q[j] - comm * q[j] * (e + x); hit[j] = 1; break
                worst = max(worst, MH[i])
        mae[j] = (worst / e - 1) * 100 * d
    return out, mae, hit
tg, _ = run(A); R = simR(D, tg); s0 = np.searchsorted(MT, t[R['eb']] + 300000); s1 = np.searchsorted(MT, t[R['xb']] + 300000); xt = t[R['xb']]
rows = []
for sl in [0, 8, 7, 6.5, 6, 5.5, 5, 4]:
    p, mae, hit = wt(s0, s1, R['dir'], R['ep'], R['xp'], R['q'], R['pnl'], MO, MH, ML, float(sl), TICK, COMM)
    cp = np.r_[0, np.cumsum(p)]; w = p[p > 0]; l = p[p < 0]
    rows.append({'anlık SL %': sl or 'yok (V5.6B)', 'net': round(p.sum()), 'PF': round(w.sum() / -l.sum(), 3), 'SL tetik': int(hit.sum()), 'aleyhe≥%7 işlem %': round((mae <= -7).mean() * 100, 1),
                 'aleyhe P90/P95/P99 %': '/'.join(f'{v:.1f}' for v in np.percentile(mae, [10, 5, 1])), 'MaxDD': round((np.maximum.accumulate(cp) - cp).max()),
                 'ilk dönem': round(p[xt < SPLIT].sum()), 'ikinci dönem': round(p[xt >= SPLIT].sum()), 'son 12 ay': round(p[xt >= t[-1] - 365 * 864e5].sum()), 'son 200 gün': round(p[xt >= t[-1] - 200 * 864e5].sum())})
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
