# (2) TV teknik derece klonu 1 s (kapanmış bar): ZEC ve BTC ikisi de ≥ eşik → short'tan vazgeç (A) / long aç (B); simetrik (ikisi de Sat → long'dan vazgeç/short) ve ölü kedi bağlamı seçenekli
# Taban: V3.1 + AKILLI POMPA (BTC 10 dk ≥ +%0,5)
exec(open('/home/claude/zecwork/akilli.py').read().split("rows = []; P = {}")[0])
import pickle; P = pickle.load(open('/home/claude/zecwork/akilli_paths.pkl', 'rb')); vp, vx, vn = P['BTC 10 dk ≥ +%0,5']
oo, hh_, ll_ = np.asarray(o, float), np.asarray(h, float), np.asarray(l, float)
rp, rx, rn = rec30(oo, hh_, ll_, cc, 0.01, vp, vx, vn, EVT, 1440, 0.08)
M = pd.concat([pd.read_parquet(f, columns=['open', 'high', 'low', 'close', 'volume']) for f in sorted(glob.glob('/home/claude/master/ZECUSDT_MASTER_1m_SYNC_*.parquet'))], ignore_index=True); M.index = ixh
ZH = pd.DataFrame({'o': M.open.resample('1h').first(), 'h': M.high.resample('1h').max(), 'l': M.low.resample('1h').min(), 'c': M.close.resample('1h').last(), 'v': M.volume.resample('1h').sum()}).dropna()
bs = pd.Series(btc, ixh); BH = pd.DataFrame({'o': bs.resample('1h').first(), 'h': bs.resample('1h').max(), 'l': bs.resample('1h').min(), 'c': bs.resample('1h').last()}).dropna(); BH['v'] = 1.0
_, _, rZ = rating(ZH); _, _, rB = rating(BH); print('derece hesaplandı', flush=True)
def m1(sr): s_ = pd.Series(sr.values, sr.index + pd.Timedelta('1h')); j = np.searchsorted(s_.index.values, ixh.values, side='right') - 1; out = s_.values[np.maximum(j, 0)]; out[j < 0] = np.nan; return out
RZ, RB = m1(rZ), m1(rB)
lo24 = pd.Series(ll_).rolling(1440, min_periods=1).min().values; hi48 = pd.Series(hh_).rolling(2880, min_periods=1).max().values; hi24 = pd.Series(hh_).rolling(1440, min_periods=1).max().values; lo48 = pd.Series(ll_).rolling(2880, min_periods=1).min().values
@njit(cache=False)
def ovR(o, h, l, c, tick, rp, rx, rn, BU, BE, mode, wait, capP):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); ovd = 0; cap = 0.0; skip = 0; nO = 0
    for i in range(1, n):
        if real != 0:
            xp = -1.0
            if ovd != 0 and real == ovd and ((real == 1 and l[i] <= cap) or (real == -1 and h[i] >= cap)): xp = (min(o[i], cap) - tick) if real == 1 else (max(o[i], cap) + tick)
            elif ovd == 0 and real == rp[i - 1] and rx[i] > 0: xp = rx[i]
            if xp > 0: rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
        v = rp[i]
        if rn[i]: skip = 0
        if mode == 2 and (BU[i] or BE[i]):
            nd = 1 if BU[i] else -1
            if ovd != nd: ovd = nd; nO += 1; tg = nd
            else: tg = real if real == nd else 0
        else:
            if ovd != 0: ovd = 0; skip = 0
            tg = v
            if (BU[i] and v < 0) or (BE[i] and v > 0): tg = 0; skip = v
            elif wait and skip != 0 and v == skip: tg = 0
            elif ovd == 0 and real != v and real != 0 and not rn[i]: tg = v
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0:
                real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
                if ovd != 0 and tg == ovd: cap = ep * (1.0 - capP * tg)
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nO
F0 = np.zeros(n, np.bool_); e0, _ = ovR(oo, hh_, ll_, cc, 0.01, rp, rx, rn, F0, F0, 1, False, 0.05); B = met(e0, 'taban'); print(f"taban (V3.1 + akıllı POMPA) {e0[-1]:.1f}")
rows = []
for tZ, tB, mode, sym, ctx, wait in itertools.product((0.1, 0.3, 0.5), (-0.1, 0.1, 0.3), (1, 2), (False, True), ('yok', 'ölü kedi %10', 'ölü kedi %15'), (False, True)):
    if mode == 2 and wait: continue
    bu = np.nan_to_num((RZ >= tZ) & (RB >= tB), nan=False); be = np.nan_to_num((RZ <= -tZ) & (RB <= -tB), nan=False) if sym else F0.copy()
    if ctx != 'yok':
        X = 0.10 if '10' in ctx else 0.15; bu = bu & (lo24 <= hi48 * (1 - X)); be = be & (hi24 >= lo48 * (1 + X)) if sym else be
    e, nO = ovR(oo, hh_, ll_, cc, 0.01, rp, rx, rn, bu.astype(np.bool_), be.astype(np.bool_), mode, wait, 0.05)
    rows.append(met(e, '', ZEC=tZ, BTC=tB, mod='A short vazgeç' if mode == 1 else 'B long aç', simetrik=sym, bağlam=ctx, bekle=wait, devir=nO))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R.to_pickle('/home/claude/zecwork/akilli2.pkl'); pd.set_option('display.width', 300)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı (≥{B['net'] * 1.06:.0f}, MaxDD≤{B['MaxDD']:.1f}, Ulcer≤{B['Ulcer']:.2f}, iki yarı): {((R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer']) & R.ikiYarı).sum()}")
print(R.groupby(['mod', 'bağlam']).agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum')).round(0).to_string())
print(R.sort_values('dnet', ascending=False)[['ZEC', 'BTC', 'mod', 'simetrik', 'bağlam', 'bekle', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']].head(8).round(2).to_string(index=False))
