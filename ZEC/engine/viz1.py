import numpy as np, pandas as pd, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
t = pd.concat([pd.read_parquet(f, columns=['time', 'close']) for f in sorted(glob.glob('/home/claude/master/*.parquet'))])
i30 = pd.to_datetime(t.time.values[::30], unit='ms'); px = pd.Series(t.close.values[::30], index=i30)
S = {'X1 (seçilen)': ('eq_X1.npy', '#c0392b', 2.0), 'X2': ('eq_X2.npy', '#e67e22', 1.1), 'H1 (Kasım)': ('eq_H1.npy', '#2471a3', 1.1), 'AKRAN-A (önceki)': ('eq_AKRAN-A.npy', '#7f8c8d', 1.0)}
E = {k: pd.Series(np.load(f), index=i30[:len(np.load(f))]) for k, (f, _, _) in S.items()}
P = pd.read_pickle('/home/claude/port/portfoy_canli_10dk_utc.pkl').resample('30min').last()
M = {  # mega2 çıktısından
 'X1 (seçilen)': dict(net=8096, n=1799, mdd=432, p99=328, cal=3.18, f99=665, lag='−%12/−%8', nov=-67, pn=45, pdd=160),
 'X2': dict(net=8006, n=1975, mdd=531, p99=340, cal=2.56, f99=846, lag='−%1/−%6', nov=-23, pn=62, pdd=154),
 'H1 (Kasım)': dict(net=7923, n=2016, mdd=635, p99=567, cal=2.12, f99=618, lag='−%23/−%22', nov=162, pn=136, pdd=143),
 'AKRAN-A (önceki)': dict(net=7430, n=1991, mdd=510, p99=373, cal=2.48, f99=820, lag='−%3/−%5', nov=-87, pn=37, pdd=203)}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(17, 11)); gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=.38, wspace=.22)
ax = fig.add_subplot(gs[0, :2])
for k, (f, col, lw) in S.items(): ax.plot(E[k].index, E[k].values, color=col, lw=lw, label=f'{k}: {E[k].iloc[-1]:,.0f} $')
mid = E['X1 (seçilen)'].index[len(E['X1 (seçilen)']) // 2]; ax.axvline(mid, ls='--', color='k', lw=.7); ax.text(mid, ax.get_ylim()[1] * .93, '  ikinci yarı (OOS) →', fontsize=8)
for y, lab in ((8000, 'hedef 8.000'), (10000, '%2000'), (15000, '%3000')): ax.axhline(y, color='#27ae60', lw=.6, ls=':'); ax.text(E['X1 (seçilen)'].index[len(E['X1 (seçilen)']) // 3], y + 120, lab, color='#27ae60', fontsize=8)
ax.set_title('Getiri eğrisi — sabit 500 $, 30 dk MTM (açık pozisyon dahil)', loc='left', fontweight='bold'); ax.legend(loc='lower right', fontsize=8); ax.set_ylabel('$')
ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
rows = [('Net $', 'net', '{:,.0f}'), ('İşlem', 'n', '{:,}'), ('MaxDD $', 'mdd', '{:.0f}'), ('DD P99 $', 'p99', '{:.0f}'), ('Calmar', 'cal', '{:.2f}'), ('12 ay ileri DD P99', 'f99', '{:.0f}'),
        ('Akran gecikmesi 1s/2s', 'lag', '{}'), ('ZEC Kasım (500 $)', 'nov', '{:+.0f}'), ('Portföy+200$ Kasım', 'pn', '{:+.0f}'), ('Portföy+200$ MaxDD', 'pdd', '{:.0f}')]
best = {'net': max, 'n': None, 'mdd': min, 'p99': min, 'cal': max, 'f99': min, 'lag': None, 'nov': max, 'pn': max, 'pdd': min}
cols = list(M); tb = [[fmt.format(M[c][k]) for c in cols] for _, k, fmt in rows]
T = ax.table(cellText=tb, rowLabels=[r[0] for r in rows], colLabels=[c.split(' ')[0] for c in cols], cellLoc='center', bbox=[0.36, 0.0, 0.64, 0.93]); T.auto_set_font_size(False); T.set_fontsize(8.5); 
for j, (_, k, _) in enumerate(rows):
    if best[k]:
        v = [M[c][k] for c in cols]; b = v.index(best[k](v)); T[j + 1, b].set_facecolor('#d5f5e3'); T[j + 1, b].set_text_props(fontweight='bold')
T[0, 0].set_facecolor('#fadbd8'); ax.set_title('Karar tablosu (yeşil = en iyi)', loc='left', fontweight='bold')
ax = fig.add_subplot(gs[1, :2])
for k, (f, col, lw) in S.items(): x = E[k]; ax.fill_between(x.index, -(x.cummax() - x), 0, color=col, alpha=.25 if k.startswith('X1') else .12, lw=0); ax.plot(x.index, -(x.cummax() - x), color=col, lw=.6)
ax.set_title('Zirveden düşüş $ — ne kadar sarsılıyor', loc='left', fontweight='bold'); ax.set_ylabel('$')
ax = fig.add_subplot(gs[1, 2]); a0 = pd.Timestamp('2026-09-25')
for k, (f, col, lw) in S.items(): x = E[k][a0:]; ax.plot(x.index, x - x.iloc[0], color=col, lw=lw, label=k.split(' ')[0])
a2 = ax.twinx(); a2.plot(px[a0:].index, px[a0:].values, color='k', lw=.6, alpha=.45); a2.set_ylabel('ZEC fiyatı', fontsize=8); a2.grid(False)
ax.axvspan(pd.Timestamp('2026-11-03'), pd.Timestamp('2026-11-12'), color='#f5b7b1', alpha=.35); ax.text(pd.Timestamp('2026-11-03'), ax.get_ylim()[0] * .9, ' portföy\n çöküşü', fontsize=7)
ax.set_title('Son 2 ay — ZEC sistemleri (siyah: ZEC fiyatı)', loc='left', fontweight='bold'); ax.legend(fontsize=7, loc='upper left'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax = fig.add_subplot(gs[2, :2]); b0 = pd.Timestamp('2026-09-01'); pp = P[b0:]; pp = pp - pp.iloc[0]
ax.plot(pp.index, pp.values, color='k', lw=1.6, label=f'Portföy 4.600 tek: {pp.iloc[-1]:+.0f}')
for k in ('X1 (seçilen)', 'H1 (Kasım)'):
    z = E[k].reindex(pp.index, method='ffill'); cc = pp + (z - z.iloc[0]) * 200 / 500; ax.plot(cc.index, cc.values, color=S[k][1], lw=1.2, label=f'Portföy + {k.split(" ")[0]} 200 $: {cc.iloc[-1]:+.0f}')
for m_, lab in ((pd.Timestamp('2026-10-01'), 'Ekim'), (pd.Timestamp('2026-11-01'), 'Kasım')): ax.axvline(m_, color='grey', lw=.6, ls='--'); ax.text(m_, ax.get_ylim()[1] * .92, ' ' + lab, fontsize=8)
ax.set_title('Portföyle birlikte (Eylül→24 Kasım) — ZEC 200 $ nominal', loc='left', fontweight='bold'); ax.legend(fontsize=8, loc='upper left'); ax.set_ylabel('$'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax = fig.add_subplot(gs[2, 2]); mm = pd.DataFrame({k: E[k].resample('MS').last().diff() for k in ('X1 (seçilen)', 'H1 (Kasım)')}).iloc[-12:]
w = 10; xs = mm.index; ax.bar(xs - pd.Timedelta(days=5), mm.iloc[:, 0], width=w, color=S['X1 (seçilen)'][1], label='X1'); ax.bar(xs + pd.Timedelta(days=5), mm.iloc[:, 1], width=w, color=S['H1 (Kasım)'][1], label='H1')
ax.axhline(0, color='k', lw=.6); ax.set_title(f'Aylık $ — son 12 ay · pozitif ay X1 {int((mm.iloc[:, 0] > 0).sum())}/12, H1 {int((mm.iloc[:, 1] > 0).sum())}/12', loc='left', fontweight='bold')
ax.legend(fontsize=8); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%y'))
fig.suptitle('ZEC 1 dk — karar panosu (araştırma motoru, TV doğrulanmadı) · veri 23.11.2026 23:59 UTC · portföy 24.11 08:30 TR', fontsize=12, fontweight='bold', x=.01, ha='left')
plt.savefig('/home/claude/zecwork/ZEC_KARAR_PANOSU.png', dpi=105, bbox_inches='tight')
