import numpy as np, v55
from short_top_lib import to5, zh, DIVh, q0, D, simR, run, evalp, m30, n
for k in (12, 24):
    z = np.nan_to_num(to5(zh(DIVh, 720))); z = np.r_[np.zeros(k), z[:-k]]
    v55.GL = (z >= 1.0).astype(np.int64); v55.GS = (z <= -1.0).astype(np.int64); e = evalp(q0); m = m30(simR(D, run(q0)[0]))
    print(f'30 günlük kapı {k//12} saat geç: net {e["net"]} OOS {e["OOS"]} L12 {e["L12"]} L200 {e["L200"]} MaxDD30 {m["MaxDD 30dk mum içi"]} Calmar30 {m["Calmar 30dk"]}')
