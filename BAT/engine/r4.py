import numpy as np, pandas as pd, json
import val
from val import battery, D0, F0
from sysB import GR
GR['PXA'] = dict(aL=[3, 3.5, 4, 4.5, 5, 5.5, 6], aS=[4, 4.5, 5, 5.5, 6, 6.5, 7], hold=[288, 360, 432, 504, 576, 648, 720], sl=[3, 4, 5, 6, 8, 10], H=[276, 282, 288, 294, 300])
C = {'V1 tepe (ilk tur)': dict(fam='PXA', H=288, aL=4, aS=5.5, hold=576, sl=5), 'V3 plato merkezi': dict(fam='PXA', H=288, aL=3.5, aS=5.5, hold=648, sl=3)}
res = {}
for lab, s in C.items():
    out, R = battery(lab, ['PXA'], dict(PXA=s)); res[lab] = (out, R)
    print(lab, {k: out[k] for k in ['net', 'n', 'pf', 'sharpe', 'sortino', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'blok', 'blok_min', 'stres', 'stres_OOS', 'monkey_max_p', 'mc_net_p5', 'mc_zarar', 'mc_dd_p95', 'mc_shuf_dd_p95', 'plato_%90', 'plato_OOSpoz', 'long', 'short', 'yr']}, flush=True)
    print(out['monkey'])
pd.to_pickle({k: v[0] for k, v in res.items()}, 'r4.pkl'); np.save('r4_R.npy', np.c_[res['V3 plato merkezi'][1]['eb'], res['V3 plato merkezi'][1]['xb'], res['V3 plato merkezi'][1]['dir'], res['V3 plato merkezi'][1]['pnl']])
