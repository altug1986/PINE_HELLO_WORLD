import numpy as np, pandas as pd, sys, itertools, json
import v55
from regime import *
a, b = float(sys.argv[1]), float(sys.argv[2]); RG, rm = reg_bars(a, b); RG = RG.astype(np.int64)
V = [-0.02, 0.0, 0.02, 0.04, 0.08, 1.0]
def ev(L, S, full=False):
    v55.TG = vote3(SCQ, ADX60, AGREE, 25.0, 0.08, 0.12, RG, np.array(L, float), np.array(S, float)); return evalp(A, full)
L = [0.0] * 5; S = [0.0] * 5; base = ev(L, S, True); best = base['sc']
for rnd in range(2):
    for r in range(5):
        cand = []
        for l, s in itertools.product(V, V):
            L2 = list(L); S2 = list(S); L2[r] = l; S2[r] = s; cand.append((ev(L2, S2)['sc'], l, s))
        sc, l, s = max(cand)
        if sc > best: best = sc; L[r] = l; S[r] = s
e = ev(L, S, True)
print(json.dumps(dict(a=a, b=b, L=L, S=S, base={k: base[k] for k in ['sc', 'net', 'OOS', 'L12', 'L200', 'calmar', 'mddc']}, new={k: e[k] for k in ['sc', 'net', 'n', 'pf', 'calmar', 'p98', 'mddc', 'IS', 'OOS', 'L12', 'L200', 'P200', 'bmin', 'bstd']}), default=float))
