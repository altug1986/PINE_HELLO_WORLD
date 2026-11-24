import sys, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load
B = load(2, src='/home/claude/master/'); c = B.close.values
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
yr = B.index.year.values
A = np.load('/home/claude/zecwork/adx_4h.npy'); adx, di = A[:, 0], A[:, 1]
R = np.where(adx >= 25, di, np.where(adx < 20, 0, 9))
f = {h: np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 for h in (120, 240, 720)}
tp = np.log(B.tt_pos_ls.values)
sig = {'ttposKontra4h': -np.sign(np.nan_to_num(tp - np.r_[np.full(120, np.nan), tp[:-120]])) * (np.abs(np.nan_to_num(tp - np.r_[np.full(120, np.nan), tp[:-120]])) > 0.03),
       'ttposKontra1g': -np.sign(np.nan_to_num(tp - np.r_[np.full(720, np.nan), tp[:-720]])) * (np.abs(np.nan_to_num(tp - np.r_[np.full(720, np.nan), tp[:-720]])) > 0.06),
       'basisKontra': M[:, ix['X17_basis']].astype(float), 'spike': M[:, ix['P09_spike']].astype(float), 'glbKontra': M[:, ix['R21_glb']].astype(float)}
qv = pd.Series(B.quote_volume.values).rolling(30).sum().values; wn = pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(30).sum().values / qv
sig['whaleTakip'] = np.sign(np.nan_to_num(wn)) * (np.abs(np.nan_to_num(wn)) > 0.05)
print('h240 bps, yön ayrı (LONG oyu / SHORT oyu) × rejim; IS/OOS; pozitif yıl sayısı')
for k, v in sig.items():
    for side in (1, -1):
        out = []
        for lab, val in (('UP', 1), ('DN', -1), ('YAT', 0), ('TÜM', None)):
            m = (v == side) & ((R == val) if val is not None else True)
            x = f[240] * side
            a = np.nanmean(x[:mid][m[:mid]]) if m[:mid].sum() > 200 else np.nan; b = np.nanmean(x[mid:][m[mid:]]) if m[mid:].sum() > 200 else np.nan
            py = sum(np.nanmean(x[(yr == y) & m]) > 0 for y in range(2021, 2027) if ((yr == y) & m).sum() > 200)
            out.append(f'{lab} {a:6.1f}/{b:6.1f} ({py})')
        print(f"{k:14s} {'L' if side > 0 else 'S'}: " + ' | '.join(out))
