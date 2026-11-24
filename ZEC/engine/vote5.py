import sys, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
w = np.zeros(len(names)); w[[ix[k] for k in ['P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'X17_basis', 'P04_htf4h']]] = 1
for H, s in ((480, 12), (240, 8), (120, 8)):
    eq, t = z02.run3(B, F, M, w, 3, -1e9, s, H); t['IS'] = t.et < mid; t['y'] = t.et.dt.year
    g = t.groupby(['side', 'IS']).pnl.agg(['sum', 'size']).round(0)
    print(f'H{H} s{s}:', {f"{'L' if k[0] > 0 else 'S'}-{'IS' if k[1] else 'OOS'}": (v['sum'], v['size']) for k, v in g.iterrows()})
    if H == 480: print(t.groupby(['y', 'side']).pnl.sum().unstack().round(0).to_string())
