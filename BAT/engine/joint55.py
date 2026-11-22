import numpy as np, pandas as pd, json, sys, time
import v55
from v55 import evalp
from scan import comp, variants, apply
from vote import vote_pos
from v54 import CFG
from v54 import comp15
X15 = comp15(1.5, 4.0)
A = dict(S=7.0, rA=4.0, L=99.0, H=576, xH=-2.0, rH=4.0, aH=20.0, hm=48, nrMin=0.1)
LP = dict(S=[6.5, 7.0, 7.5, 8.0], H=[504, 576, 648], xH=[-3.0, -2.0, -1.0], aH=[15.0, 20.0, 25.0], nrMin=[0.0, 0.1, 0.2], hm=[0, 16, 32, 48], rA=[3.0, 4.0, 5.0])
VAR = list(variants()); G = {}
for lab, ch in VAR:
    key = (ch[0], ch[1] if ch[0] != 'p' else tuple(ch[1])); G.setdefault(str(key), []).append(ch)
G = list(G.values()); rng = np.random.default_rng(int(sys.argv[1])); rows = []; t0 = time.time()
def tgt(cfg, w15):
    X = np.vstack([np.stack([comp(j, cfg['p'][j]) for j in range(1, 15)]), X15[None, :]])
    return vote_pos(X, np.array(cfg['w'] + [w15], float), cfg['on'], cfg['off'], 1)[0]
first = True
while time.time() - t0 < float(sys.argv[2]):
    cfg = json.loads(json.dumps(CFG)); cfg['p'] = {int(k): v for k, v in cfg['p'].items()}; chs = []; p = dict(A); w15 = 1.0
    if not first:
        for g in G:
            if rng.random() < 0.08: ch = g[rng.integers(len(g))]; cfg = apply(cfg, ch); chs.append(ch)
        for k, v in LP.items():
            if rng.random() < 0.3: p[k] = v[rng.integers(len(v))]
        if rng.random() < 0.2: w15 = [0.0, 0.5, 2.0][rng.integers(3)]
    first = False
    v55.TG = tgt(cfg, w15); e = evalp(p); rows.append(dict(chs=json.dumps(chs), w15=w15, **{k: p[k] for k in LP}, **e))
d = pd.DataFrame(rows); d.to_pickle(f'j55_{sys.argv[1]}.pkl'); print(len(d), 'varyant', f'{time.time()-t0:.0f}s')
