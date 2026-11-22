import numpy as np
import v55
from v54 import CFG, comp15
from scan import comp
from vote import vote_pos
from rg55 import vote2, AGREE, ADX60
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
X15 = comp15(1.5, 4.0); p11 = dict(CFG['p'][11]); p11['x'] = 2.5
_X = np.vstack([np.stack([comp(j, p11 if j == 11 else CFG['p'][j]) for j in range(1, 15)]), X15[None, :]])
SCQ = vote_pos(_X, np.array(CFG['w'] + [1.0], float), CFG['on'], CFG['off'], 1)[1]
TGQ = vote2(SCQ, ADX60, AGREE, 0.0, 25.0, 0.08, 0.08, 0.12, 0.08, 0.0)
v55.TG = TGQ
