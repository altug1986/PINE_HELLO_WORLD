import sys, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02, regime
B = load(2, src='/home/claude/master/'); c = B.close.values; n = len(c)
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json'))
mid = np.searchsorted(B.index.values, (B.index[0] + (B.index[-1] - B.index[0]) / 2).to_datetime64())
fw = {h: np.r_[c[h:] / c[:-h] - 1, np.full(h, np.nan)] * 1e4 for h in (60, 240)}
# sürekli öncü adaylar (balina, OI, emir defteri, top trader) → işaret oyu
qv = pd.Series(B.quote_volume.values).rolling(30).sum().values
ext = {
 'W_whale30': np.sign(np.nan_to_num(pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(30).sum().values / qv)) * (np.abs(np.nan_to_num(pd.Series(B.wh_buy_q50k.values - B.wh_sell_q50k.values).rolling(30).sum().values / qv)) > 0.05),
 'W_bigprint': np.sign(np.nan_to_num(pd.Series(B.wh_buy_q200k.values - B.wh_sell_q200k.values).rolling(15).sum().values)),
 'O_oiUpPxUp': None, 'B_book': None}
oi = B.oi_zec.values; d1 = (oi / np.r_[np.full(30, np.nan), oi[:-30]] - 1) * 100; d4 = (oi / np.r_[np.full(120, np.nan), oi[:-120]] - 1) * 100
px1 = (c / np.r_[np.full(30, np.nan), c[:-30]] - 1) * 100; px4 = (c / np.r_[np.full(120, np.nan), c[:-120]] - 1) * 100
ext['O_oiUpPxUp'] = np.where((d1 > 1) & (px1 > 0), 1, np.where((d1 > 1) & (px1 < 0), -1, 0))      # yeni para yönü
ext['O_oi4hBuild'] = np.where((d4 > 3) & (px4 > 0), 1, np.where((d4 > 3) & (px4 < 0), -1, 0))
ext['O_oi4hFlush'] = np.where((d4 < -3) & (px4 < 0), 1, np.where((d4 < -3) & (px4 > 0), -1, 0))   # tasfiye → dönüş
bi = ((B.book_m1 - B.book_p1) / (B.book_m1 + B.book_p1)).values; ext['B_book'] = np.sign(np.nan_to_num(bi)) * (np.abs(np.nan_to_num(bi)) > 0.2)
tp = np.log(B.tt_pos_ls.values); ext['T_ttpos4h'] = np.sign(np.nan_to_num(tp - np.r_[np.full(120, np.nan), tp[:-120]])) * (np.abs(np.nan_to_num(tp - np.r_[np.full(120, np.nan), tp[:-120]])) > 0.03)
allv = {k: M[:, j].astype(float) for j, k in enumerate(names)}; allv.update({k: np.asarray(v, float) for k, v in ext.items()})
for rule in ('1h', '4h'):
    adx, di = regime.regime(B, rule)
    np.save(f'/home/claude/zecwork/adx_{rule}.npy', np.c_[adx, di])
    R = np.where(adx >= 25, di, np.where(adx < 20, 0, np.nan))     # +1 yukarı trend, −1 aşağı trend, 0 yatay, NaN geçiş
    share = {k: round(float(np.mean(R == v)) * 100, 1) for k, v in (('UP', 1), ('DN', -1), ('YATAY', 0))}
    print(f'\n== ADX {rule} rejim payı %', share, '| h240 bps oy yönünde, IS/OOS; [n aktif bar bin]')
    rows = []
    for k, v in allv.items():
        act = v != 0; row = {'bileşen': k}
        for lab, val in (('UP', 1), ('DN', -1), ('YATAY', 0)):
            m = act & (R == val); f = fw[240] * v
            a = np.nanmean(f[:mid][m[:mid]]) if m[:mid].sum() > 300 else np.nan; b = np.nanmean(f[mid:][m[mid:]]) if m[mid:].sum() > 300 else np.nan
            row[lab] = f'{a:6.1f}/{b:6.1f} [{m.sum()//1000}]'
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    # yön ön koşulu: rejimde trend yönünde tutmanın getirisi
    for lab, val in (('UP', 1), ('DN', -1)):
        m = R == val; f = fw[240] * val
        print(f'  rejim yönünde tut ({lab}): h240 IS {np.nanmean(f[:mid][m[:mid]]):.1f} OOS {np.nanmean(f[mid:][m[mid:]]):.1f} bps')
