import sys, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load
B = load(2, src='/home/claude/master/'); c = B.close.values
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json'))
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
yr = B.index.year.values
print('bileşen     aktif%  | ileri getiri bps (oy yönünde): h15 IS/OOS | h60 IS/OOS | h240 IS/OOS | h60 pozitif yıl')
for j, k in enumerate(names):
    v = M[:, j].astype(float); act = v != 0; out = []
    for h in (15, 60, 240):
        f = np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 * v
        a = np.nanmean(f[:mid][act[:mid]]); b = np.nanmean(f[mid:][act[mid:]]); out.append(f'{a:6.1f}/{b:6.1f}')
        if h == 60:
            py = sum(np.nanmean(f[(yr == y) & act]) > 0 for y in range(2021, 2027) if ((yr == y) & act).sum() > 50)
    print(f'{k:11s} {act.mean()*100:5.1f} | ' + ' | '.join(out) + f' | {py}/6')
