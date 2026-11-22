import numpy as np, pandas as pd
from btcx import D, c, t, run, report, ent, cap, b24, b1
from rev25 import feats, apply25
s = ((b24 <= -10) & (b1 >= 0)).astype(np.int64); side = np.where(cap != 0, cap, ent(s, c, 288, 3.0)); rows = []
for w, wt in [('tam', False), ('tv', True)]:
    F = feats(D, wt); core = run(D, whale=w, ovr=side); rows.append(report(D, core, f'V6.9 · {w}'))
    for lab, k in [('+ yeniden giriş', (1, 0, 0)), ('+ derin trend bekletme', (0, 0, 1)), ('+ mikro bekletme', (0, 1, 0)), ('+ REV-25 hepsi', (1, 1, 1))]:
        rows.append(report(D, apply25(D, core, F, *k), f'V6.9 {lab} · {w}'))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
