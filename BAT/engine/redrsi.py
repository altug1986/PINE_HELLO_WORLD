import numpy as np, pandas as pd, itertools
import base56b, v55
from base56b import A
from v55 import evalp, run, simR, D, c, t, n
from feat import hk15, k15, c15
from eng import rsi, htf_map
from teco import daily, mask_from, teco
R15 = np.nan_to_num(htf_map(rsi(c15, 14), hk15, k15), nan=50.0)           # önceki tamamlanmış 15 dk mum RSI14
dk = t // 86400000; ud, st = np.unique(dk, return_index=True); en = np.r_[st[1:], n]; red = c[en - 1] < D['o'][st]; grn = c[en - 1] > D['o'][st]
di = np.searchsorted(ud, dk); prevRed = np.where(di > 0, red[np.maximum(di - 1, 0)], False); prevGrn = np.where(di > 0, grn[np.maximum(di - 1, 0)], False)
hr = (t // 3600000) % 24
tg0, _ = run(A); R0 = simR(D, tg0); M = mask_from(daily(R0)); print('maske:', int(M.sum()), 'gün (yılda en çok 3 × 7 gün, bazdan sabit)')
def go(GL, GS, lab):
    v55.GL = GL.astype(np.int64); v55.GS = GS.astype(np.int64); e = evalp(A); tg, _ = run(A); R = simR(D, tg); f, co, ta, l12, l12c = teco(R, M)
    return dict(t=lab, net=e['net'], n=e['n'], pf=e['pf'], calmar=e['calmar'], p98=e['p98'], mddc=e['mddc'], IS=e['IS'], OOS=e['OOS'], L12=e['L12'], L12core=l12c,
                coreNet=co['net'], coreSh=co['sharpe'], coreSo=co['sortino'], coreCal=co['calmar'], coreCVw=co['cvW'], coreUW=co['uw'], coreP95=co['p95'], coreP99=co['p99'], tail=ta)
z = np.zeros(n, bool)
if __name__ == '__main__':
  rows = [go(z, z, 'BAZ V5.6B')]
  for W, th in itertools.product([12, 6, 18, 24], [60, 55, 65, 50]):
    rows.append(go(prevRed & (hr < W) & (R15 <= th), z, f'long: kırmızı gün sonrası {W}s RSI15>{th}'))
  for W, th in [(12, 60), (12, 55)]:
    rows.append(go(prevRed & (hr < W) & (R15 <= th), prevGrn & (hr < W) & (R15 >= 100 - th), f'iki yön simetrik {W}s {th}/{100-th}'))
  d = pd.DataFrame(rows); d.to_pickle('redrsi.pkl'); pd.set_option('display.width', 300); print(d.to_string(index=False))
