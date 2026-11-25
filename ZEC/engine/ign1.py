"""R5 stres teşhisi + IGNITION (hareket başlangıcı) olay çalışması, 1 dk, nedensel."""
import sys, itertools; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z01, z03
exec(open('/home/claude/zecwork/bob1.py').read().split("blk = np.searchsorted")[0])
L5, S5 = select(15, 240); aL5 = (L5 >= 2); aS5 = (S5 >= 3) & ~aL5
eq, T = z03.sim7(o, h, l, c, a, aL5, aS5, 16.0, 1920, z, 8.0, 12.0, 240, spk, 60); e = pd.Series(eq, index=idx)
np.save('/home/claude/zecwork/r5_eq.npy', eq)
def win(a_, b_): x = e.loc[a_:b_]; return x.iloc[-1] - x.iloc[0], (x.cummax() - x).max()
for lab, a_, b_ in (('son 2 ay', '2026-09-24', '2026-11-24'), ('son 3 hafta', '2026-11-03', '2026-11-24'), ('23 Kas', '2026-11-23 00:00', '2026-11-24'), ('2022-06→2023-03', '2022-06-01', '2023-04-01')):
    p_, d_ = win(a_, b_); print(f'R5 {lab}: {p_:+.0f} $ (%{p_/5:+.0f}) · içindeki DD {d_:.0f}')
# --- IGNITION özellikleri
B_ = load(1, src='/home/claude/master/') if False else None
from zec_data import master
X = master('/home/claude/master/')[['volume', 'taker_delta_base', 'wh_buy_q50k', 'wh_sell_q50k', 'quote_volume', 'oi_zec']]
v = X.volume.values.astype(float); td = X.taker_delta_base.values.astype(float); wn = (X.wh_buy_q50k.values - X.wh_sell_q50k.values).astype(float); qv = X.quote_volume.values.astype(float)
oi = np.log(X.oi_zec.values.astype(float)); del X
roll = lambda x, w, f: getattr(pd.Series(x).rolling(w, min_periods=max(2, w // 3)), f)().values
lc = np.log(c); r1 = np.r_[0, np.diff(lc)]; vol60 = roll(r1, 60, 'std'); vol1d = roll(r1, 1440, 'std'); vr = roll(vol60 / vol1d, 60, 'mean')
fl5 = roll(td, 5, 'sum') / roll(v, 5, 'sum'); wh10 = roll(wn, 10, 'sum') / roll(qv, 10, 'sum'); oi15 = (oi - sh(oi, 15)) * 100
np.savez('/home/claude/zecwork/ign_feat.npz', vr=vr, fl5=fl5, wh10=wh10, oi15=oi15)
COST = 8.3; yr = idx.year.values
fw = {H: (np.r_[c[H:] / c[:-H], np.full(H, np.nan)] - 1) * 1e4 for H in (30, 120, 480)}
rows = []
for N, vq, fx, oiReq in itertools.product([60, 120, 240], [0.7, 0.9, 99], [0.0, 0.2, 0.35], [False, True]):
    HH = sh(roll(h, N, 'max'), 1); LL = sh(roll(l, N, 'min'), 1)
    base = (vr < vq) if vq < 99 else np.ones(len(c), bool)
    up = base & (c > HH) & (sh(c, 1) <= sh(HH, 1)) & (fl5 >= fx) & (wh10 >= 0)
    dn = base & (c < LL) & (sh(c, 1) >= sh(LL, 1)) & (fl5 <= -fx) & (wh10 <= 0)
    if oiReq: up &= oi15 > 0; dn &= oi15 > 0
    for side, m in ((1, up), (-1, dn)):
        i = np.where(np.nan_to_num(m))[0]; keep = []; last = -10**9
        for j in i:
            if j - last >= 60: keep.append(j); last = j
        i = np.array(keep, int); i = i[(i > 3000) & (i < len(c) - 500)]
        if len(i) < 60: continue
        r = dict(N=N, sıkışma=vq, akış=fx, OI=oiReq, yön='L' if side > 0 else 'S', n=len(i))
        for H in (30, 120, 480):
            x = fw[H][i] * side - COST; r[f'h{H}'] = np.nanmean(x); r[f'h{H}IS'] = np.nanmean(x[i < mid]); r[f'h{H}OOS'] = np.nanmean(x[i >= mid])
        r['yıl+'] = int(sum(np.nanmean(fw[120][i[yr[i] == y]] * side) > COST for y in range(2021, 2027) if (yr[i] == y).sum() >= 5))
        r['23Kas'] = int(((idx[i] >= '2026-11-23 13:30') & (idx[i] <= '2026-11-23 14:30')).sum())
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/IGNITION_SCAN.csv', index=False); pd.set_option('display.width', 250)
G = R[(R.h120IS > 0) & (R.h120OOS > 0)].sort_values('h120', ascending=False)
print(f'\nIGNITION: iki yarıda net pozitif (h120): {len(G)} / {len(R)}'); print(G.head(16).round(1).to_string(index=False))
