# 4 PARİTE 5 dk BİRLEŞİK ADX (ZEC, BTC, XMR, ETH) × saat olasılık çarpanı → REGÜLATİF DİNLENME BÖLGESİ (V3.5 üstüne)
exec(open('/home/claude/zecwork/bekle5.py').read().split("out = []")[0])
import peerfeat, itertools
P_ = '/home/claude/peer/'; S2 = '/home/claude/pine_hello_world/data/sync2611/'
def dmi(H, L, C):
    pc = np.r_[C[0], C[:-1]]; tr = np.maximum(H - L, np.maximum(abs(H - pc), abs(L - pc))); up = np.r_[0, np.diff(H)]; dn = np.r_[0, -np.diff(L)]; at = rma(tr, 14)
    p = 100 * rma(np.where((up > dn) & (up > 0), up, 0), 14) / at; m = 100 * rma(np.where((dn > up) & (dn > 0), dn, 0), 14) / at; return rma(100 * abs(p - m) / np.where(p + m == 0, 1, p + m), 14)
def to1m(D5):   # kapanmış 5 dk bar (Pine: request.security("5", adx[1], lookahead_on))
    D5 = D5.dropna(); ax = dmi(D5.high.values, D5.low.values, D5.close.values); ct = D5.index.values + np.timedelta64(5, 'm')
    j = np.searchsorted(ct, ixh.values, side='right') - 1; return np.where(j >= 0, ax[np.maximum(j, 0)], np.nan)
r5 = lambda d: d.resample('5min').agg({'high': 'max', 'low': 'min', 'close': 'last'})
AX = {'ZEC': to1m(r5(pd.DataFrame({'high': h_, 'low': l_, 'close': cc}, ixh)))}
AX['BTC'] = to1m(r5(peerfeat.rd([P_ + 'BTCUSDT_PERP_1m_2019_20261115.csv.gz'], 'time', 'BTCUSDT')))
AX['XMR'] = to1m(r5(peerfeat.rd([P_ + 'XMRUSDT_1m_FULL_THROUGH_20261116.csv.gz'], 'time', 'XMRUSDT')))
E5 = pd.read_csv('/home/claude/pine_hello_world/data/ETHUSDT_5m_UTC_20210101_THROUGH_20261124.csv.gz'); E5.index = pd.to_datetime(E5.time, unit='ms')
Ep = pd.read_csv(S2 + 'ETHUSDT_SPOT_PROXY_1m_UTC_20261125_20261126.csv.gz'); Ep.index = pd.to_datetime(Ep.time, unit='ms')
AX['ETH'] = to1m(pd.concat([E5[['high', 'low', 'close']], r5(Ep[['high', 'low', 'close']])]).pipe(lambda d: d[~d.index.duplicated()]).sort_index())
print('ADX kapsama %', {k: round(float(np.isfinite(v).mean() * 100), 1) for k, v in AX.items()}, '| korelasyon ZEC–BTC', round(float(pd.Series(AX['ZEC']).corr(pd.Series(AX['BTC']))), 2))
e_, TT = comb3T(o_, h_, l_, cc, 0.01, rp, rx, S9, A9, 38.0, 0.05); hr = ixh.hour.values
ti = TT[TT[:, 0] < midI]; wr = pd.Series(ti[:, 5] > 0).groupby(hr[ti[:, 0].astype(int)]).mean(); fH = (wr / (ti[:, 5] > 0).mean()).reindex(range(24)).fillna(1.0).values
print('saat çarpanı (IS kazanma oranı / ort.) min/maks', round(fH.min(), 2), round(fH.max(), 2), '| en zayıf saatler UTC', list(np.argsort(fH)[:4]))
@njit(cache=False)
def ovR(o, c, tick, fp, fx, rest, mode):
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); skip = 0; nb = 0
    for i in range(1, n):
        if real != 0 and real == fp[i - 1] and fx[i] > 0: rz += q * (fx[i] - ep) * real - 0.0004 * q * (ep + fx[i]); real = 0
        tg = int(fp[i])
        if skip != 0 and tg != skip: skip = 0
        if rest[i]:
            if real != 0 and tg == real:
                if mode == 1 and (c[i] - ep) * real > 0: tg = 0                       # dinlenmede kârdaki işlemi kilitle
            else:
                if tg != 0: skip = tg; nb += 1
                tg = 0
        elif mode == 2 and real == 0 and tg != 0 and tg == skip: tg = 0               # dinlenme bitince geç giriş yok, yeni sinyal bekle
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nb
SETS = [('ZEC',), ('BTC',), ('XMR',), ('ETH',), ('ZEC', 'BTC'), ('ZEC', 'BTC', 'ETH'), ('ZEC', 'BTC', 'XMR', 'ETH')]
out = []; isM = np.arange(n) < midI
for st, ag, tm in itertools.product(SETS, ('ort', 'maks'), (0, 1)):
    M = np.vstack([AX[k] for k in st]); sc = (np.nanmean(M, 0) if ag == 'ort' else np.nanmax(M, 0)) * (fH[hr] if tm else 1.0)
    for qq in (0.1, 0.2, 0.3, 0.4):
        thr = np.nanquantile(sc[isM], qq); rest = np.nan_to_num(sc, nan=99.0) < thr
        for mode in (0, 1, 2):
            e, nb = ovR(o_, cc, 0.01, fp, fx, rest, mode); m = met(e, '')
            out.append(('+'.join(st), ag, tm, qq, ['yeni giriş yok', '+kârı kilitle', 'geç giriş yok'][mode], round(float(thr), 1), m['net'] - B5['net'], m['IS'] - B5['IS'], m['OOS'] - B5['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], nb, rest.mean()))
d = pd.DataFrame(out, columns=['set', 'agg', 'saat', 'q', 'mod', 'eşik', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'engel', 'dinlenme%']); d.to_csv('adx4.csv', index=False)
print('baz', round(B5['net'], 1), 'MaxDD', round(B5['MaxDD'], 1), '| varyant', len(d), '| pozitif', int((d.dNet > 0).sum()), '| iki yarı +', int(((d.dIS > 0) & (d.dOOS > 0)).sum()),
      '| kapı (≥+%6, MaxDD ≤ baz, iki yarı +)', int(((d.dNet >= 0.06 * B5['net']) & (d.MaxDD <= B5['MaxDD']) & (d.dIS > 0) & (d.dOOS > 0)).sum()))
pd.set_option('display.width', 250)
for k in ('set', 'mod', 'q', 'saat', 'agg'): print(k, d.groupby(k).dNet.median().round(0).to_dict())
print('--- IS en iyi 8'); print(d.sort_values('dIS', ascending=False).head(8).round(1).to_string(index=False))
print('--- MaxDD en düşük 5 (iki yarı +)'); print(d[(d.dIS > 0) & (d.dOOS > 0)].sort_values('MaxDD').head(5).round(1).to_string(index=False))
