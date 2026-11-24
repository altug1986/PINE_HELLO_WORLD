import sys, itertools, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01, z02
B = load(2, src='/home/claude/master/'); F = z01.prep(B); n = len(B); c = F['c']; a = F['atr']
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
# Pine-birebir oylar (BB popülasyon std = ta.stdev)
mu = pd.Series(c).rolling(120).mean().values; sd = pd.Series(c).rolling(120).std(ddof=0).values
bb = np.where(c < mu - 2.5 * sd, 1, np.where(c > mu + 2.5 * sd, -1, 0))
M = np.load('/home/claude/zecwork/votes.npy'); names = json.load(open('/home/claude/zecwork/vote_names.json')); ix = {k: j for j, k in enumerate(names)}
V = {k: M[:, ix[k]].astype(float) for k in ['P07_rsi', 'P08_zema', 'P09_spike', 'P12_oiLiq', 'P04_htf4h', 'X17_basis']}; V['P10_bb'] = bb
def atr4(H, k=14):
    c1 = H.close.shift(1); tr = np.maximum(H.high - H.low, np.maximum((H.high - c1).abs(), (H.low - c1).abs())); return tr.ewm(alpha=1 / k, adjust=False).mean().values
A4 = z02.htf(B, '4h', atr4); z = np.zeros(n, np.bool_)
def run(N=10, k=4, keys=('P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'P04_htf4h'), Tv=3, ts=4, tH=480):
    hh = z02.htf(B, '4h', lambda H: H.high.rolling(N).max().values)
    eL = np.nan_to_num(c > hh).astype(np.bool_); S = sum(V[k_] for k_ in keys); tL = (S >= Tv).astype(np.bool_)
    eq, T = z02.sim4(F['o'], F['h'], F['l'], c, a, eL, z, A4 * k, tL, z, float(ts), int(tH), 99.0)
    t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own']); t['et'] = B.index[t.ei.astype(int)]
    return pd.Series(eq, index=B.index), t
for lab, kw in (('X17 dahil (7 oy)', dict(keys=('P07_rsi', 'P08_zema', 'P09_spike', 'P10_bb', 'P12_oiLiq', 'P04_htf4h', 'X17_basis'))), ('Pine-birebir (6 oy)', {})):
    e, t = run(**kw); f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None); s = z01.sc4(e)
    print(f"{lab}: net {f['net']:.0f} (%{f['pct']:.0f}) n {f['n']} PF {f['PF']:.2f} win {f['win']:.0f}% | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} Sortino {f['Sortino']:.2f} Ulcer {f['Ulcer']:.0f} | SC4 {s['SC4']:.3f} A95 {s['A95']:.0f}g | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 {f['maeP90']:.1f}% max {f['maeMax']:.1f}% | IS {A['net']:.0f} OOS {Bm['net']:.0f} | +ay %{f['posM']:.0f} | trend {t[t.own==1].pnl.sum():.0f} taktik {t[t.own==2].pnl.sum():.0f} | top3 %{t.pnl.nlargest(3).sum()/f['net']*100:.0f}")
t['y'] = t.et.dt.year; print(t.groupby('y').pnl.agg(['sum', 'size']).round(0).T.to_string())
R = []
for N, k, Tv, ts, tH in itertools.product([5, 10, 20], [3, 4, 5], [2, 3, 4], [4, 8], [240, 480]):
    e, tt = run(N, k, Tv=Tv, ts=ts, tH=tH); R.append(e.iloc[-1])
R = np.array(R); print('komşu', len(R), 'pozitif %', (R > 0).mean() * 100, 'medyan', np.median(R).round(0), 'P10', np.percentile(R, 10).round(0), 'maks', R.max().round(0))
e.to_frame('eq')  # son
e0, t0 = run(); t0.to_csv('/home/claude/pine_hello_world/ZEC/research/V0.2_trades_engine.csv', index=False)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m30 = e0.resample('30min').last().dropna(); dd = m30.cummax() - m30
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
ax[0].plot(m30.index, m30.values, '#1f6feb', lw=1); ax[0].set_title('CYBORG | ZEC 2M V0.2 — 30 dk MTM equity (USDT, sabit 500 giriş, araştırma motoru)')
ax[0].axvline(mid, color='#888', ls='--', lw=.8); ax[1].fill_between(dd.index, -dd.values, color='#d1242f', lw=0); ax[1].set_ylabel('MTM DD $')
for x in ax: x.grid(alpha=.3)
plt.tight_layout(); plt.savefig('/home/claude/pine_hello_world/ZEC/research/V0.2_equity_dd.png', dpi=110)
