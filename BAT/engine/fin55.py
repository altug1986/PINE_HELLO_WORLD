import numpy as np, pandas as pd, json, itertools
import v55
from v55 import evalp
from scan import comp, apply
from vote import vote_pos
from v54 import CFG, comp15
X15 = comp15(1.5, 4.0); TG0 = v55.TG.copy()
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
def tgt(chs):
    cfg = json.loads(json.dumps(CFG)); cfg['p'] = {int(k): v for k, v in cfg['p'].items()}
    for ch in chs: cfg = apply(cfg, (ch[0], tuple(ch[1]) if isinstance(ch[1], list) else ch[1], ch[2]))
    X = np.vstack([np.stack([comp(j, cfg['p'][j]) for j in range(1, 15)]), X15[None, :]]); return vote_pos(X, np.array(cfg['w'] + [1.0], float), cfg['on'], cfg['off'], 1)[0]
pd.set_option('display.width', 250); K = ['sc', 'h0', 'h1', 'net', 'n', 'pf', 'calmar', 'p98', 'mddc', 'IS', 'OOS', 'last12', 'bmin', 'bstd', 'bneg']
rows = []
for beP, beL, beT in itertools.product([2.0, 3.0, 4.0, 5.0, 6.0], [-3.0, -2.0, -1.0, 0.0, 1.0], [0]):
    rows.append(dict(t=f'BE {beP}/{beL}', **evalp(dict(A, beP=beP, beL=beL, beT=beT))))
d = pd.DataFrame(rows); print('KÂRI KORUMA (başa baş stop) en iyi 4 / 25:'); print(d.sort_values('sc', ascending=False).head(4)[['t'] + K].to_string(index=False))
rows = []
for lab, chs in [('A', []), ('A + BTC lider %2.5', [['p', [11, 'x'], 2.5]]), ('A + BTC lider %2.25', [['p', [11, 'x'], 2.25]]), ('A + BTC lider %3', [['p', [11, 'x'], 3.0]]),
                 ('A + giriş 0.04 + BTC lider %2.5', [['on', None, 0.04], ['p', [11, 'x'], 2.5]]), ('A + giriş 0.06 + BTC lider %2.5', [['on', None, 0.06], ['p', [11, 'x'], 2.5]])]:
    v55.TG = tgt(chs); rows.append(dict(t=lab, **evalp(A)))
print(pd.DataFrame(rows)[['t'] + K].to_string(index=False))
