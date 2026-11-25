import sys; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, z01, z03
exec(open('/home/claude/zecwork/bob1.py').read().split("blk = np.searchsorted")[0])
L, S = select(15, 240); aL = (L >= 2); aS = (S >= 3) & ~aL
eq, T = z03.sim7(o, h, l, c, a, aL, aS, 16.0, 1920, z, 8.0, 12.0, 240, spk, 60)
t = pd.DataFrame(T, columns=['ei', 'xi', 'side', 'ep', 'xp', 'pnl', 'mae', 'own', 'guard']); t['et'] = idx[t.ei.astype(int)]
e = pd.Series(eq, index=idx); midT = idx[mid]
f = z01.metrics(e, t); A = z01.metrics(e, t, None, midT); Bm = z01.metrics(e, t, midT, None); sc = z01.sc4(e)
b2 = e.resample('14D').last().diff().dropna()
print(f"R5 BoB: net {f['net']:.0f} (%{f['pct']:.0f}) n {f['n']} PF {f['PF']:.2f} kazanma %{f['win']:.0f} işlem başı {f['bps']:.0f} bps | MaxDD {f['MaxDD']:.0f} P95/98/99 {f['P95']:.0f}/{f['P98']:.0f}/{f['P99']:.0f} | Calmar {f['Calmar']:.2f} Sharpe {f['Sharpe']:.2f} Sortino {f['Sortino']:.2f} | SC4 {sc['SC4']:.3f} A95 {sc['A95']:.0f}g | σ10dk {f['s10']:.2f} σ1s {f['s1h']:.2f} | MAE P90 %{f['maeP90']:.1f} P95 %{np.percentile(t.mae,95)*100:.1f} P99 %{np.percentile(t.mae,99)*100:.1f} maks %{f['maeMax']:.1f} | IS {A['net']:.0f} OOS {Bm['net']:.0f}")
print(f"2 haftalık bloklar ({len(b2)}): medyan {b2.median():.1f} $ · ortalama {b2.mean():.1f} · pozitif %{(b2>0).mean()*100:.0f} · en kötü {b2.min():.0f} · P10 {b2.quantile(.1):.0f} · P90 {b2.quantile(.9):.0f}")
print('yıllar', t.groupby(t.et.dt.year).pnl.sum().round(0).astype(int).to_dict(), '| long', round(t[t.side>0].pnl.sum()), 'short', round(t[t.side<0].pnl.sum()))
w = e.loc['2022-06-01':'2023-03-31']; print(f"zayıf dönem 2022-06 → 2023-03: {w.iloc[-1]-w.iloc[0]:+.0f} $ · içindeki MaxDD {((w.cummax()-w).max()):.0f}")
R = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/BOB_1DK_2HAFTA.csv')
nb = R[(R.thr.isin([10, 15, 20])) & (R.hz == 240) & (R.kL == 2) & (R.kS.isin([3, 4])) & (R.ts.isin([11, 16, 24])) & (R.tH.isin([960, 1920]))]
print(f'komşular ({len(nb)}): net medyan {nb.net.median():.0f} min {nb.net.min():.0f} maks {nb.net.max():.0f} · zayıf dönem pozitif {int((nb.zayıf>0).sum())}/{len(nb)} · iki yarı pozitif {int(((nb.IS>0)&(nb.OOS>0)).sum())}/{len(nb)}')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m30 = e.resample('30min').last().dropna(); dd = m30.cummax() - m30
x = np.arange(len(m30)); k_, b_ = np.polyfit(x, m30.values, 1)
fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=False, gridspec_kw={'height_ratios': [3, 1.1, 1.3]})
ax[0].plot(m30.index, m30.values, '#1f6feb', lw=1, label='R5 (1 dk, BoB)'); ax[0].plot(m30.index, k_ * x + b_, 'r', lw=1.5, label='doğrusal hedef çizgisi')
ax[0].axvspan(pd.Timestamp('2022-06-01'), pd.Timestamp('2023-04-01'), color='orange', alpha=.1); ax[0].axvline(midT, color='#888', ls='--', lw=.8)
ax[0].set_title('ZEC 1M R5 — 30 dk MTM equity (USDT, sabit 500, araştırma motoru) · turuncu = 2022-06→2023-03'); ax[0].legend(loc='upper left')
ax[1].fill_between(dd.index, -dd.values, color='#d1242f', lw=0); ax[1].set_ylabel('MTM DD $'); ax[1].set_xlim(ax[0].get_xlim())
ax[2].bar(b2.index, b2.values, width=12, color=np.where(b2.values > 0, '#1f6feb', '#d1242f')); ax[2].axhline(b2.median(), color='k', lw=.8, ls='--'); ax[2].set_ylabel('2 hafta $'); ax[2].set_xlim(ax[0].get_xlim())
for x_ in ax: x_.grid(alpha=.3)
plt.tight_layout(); plt.savefig('/home/claude/pine_hello_world/ZEC/research/R5_1dk_BoB.png', dpi=110)
t.to_csv('/home/claude/pine_hello_world/ZEC/research/R5_1dk_trades_engine.csv', index=False)
