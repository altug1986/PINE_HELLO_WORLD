import numpy as np, pandas as pd
import search57 as S
from scan import comp
from vote import vote_pos
from v54 import CFG
import whsig as W
p11 = dict(CFG['p'][11]); p11['x'] = 2.5
M = np.vstack([np.stack([comp(j, p11 if j == 11 else CFG['p'][j]) for j in range(1, 15)]), S.X15[None, :]])
def sc(off): X = M.copy(); X[[j - 1 for j in off]] = 0; return vote_pos(X, S.W, CFG['on'], CFG['off'], 1)[1]
print('ağırlıklar 7/8:', S.W[6], S.W[7], '| toplam', S.W.sum(), '| regresyon farkı', float(np.abs(sc([]) - W.SC).max()))
rows = []
for lab, off in [('V6.6 (OI açık)', []), ('7 likidasyon (OI) kapalı', [7]), ('8 OI kurulum kapalı', [8]), ('ikisi de kapalı', [7, 8])]:
    W.SC = sc(off); rows.append(W.ev(lab=lab)); W.SC = sc([])
    rows[-1]['tetik bar %'] = round(np.mean(np.any(M[[j - 1 for j in (off or [7, 8])]] != 0, axis=0)) * 100, 1)
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
