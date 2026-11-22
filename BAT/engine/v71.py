import numpy as np, pandas as pd
from btcx import D, c, t, run, report, trades, ent, b24, b1
from edge2 import sig
from rev25 import feats, apply25
F = feats(D, False); deep = F['deep']
capS = sig('Kapitülasyon', (5, 4)); btcS = ((b24 <= -10) & (b1 >= 0)).astype(np.int64)
def side(gate):
    cs = np.where(gate & (deep == -1), 0, capS).astype(np.int64); bs = np.where(gate & (deep == -1), 0, btcS).astype(np.int64)
    cap = ent(cs, c, 144, 5.0); return np.where(cap != 0, cap, ent(bs, c, 288, 3.0))
rows = []
for g, gl in [(False, ''), (True, ' · yan motor derin düşüşte kapalı')]:
    core = run(D, ovr=side(g))
    rows.append(report(D, apply25(D, core, F, 0, 1, 0), 'V7.0' + gl))
    rows.append(report(D, apply25(D, core, F, 1, 1, 1), 'V7.0 + V6.7 trend taşıma' + gl))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
