import sys, itertools, json; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd
from zec_data import load; import z01
B = load(2, src='/home/claude/master/'); F = z01.prep(B)
mid = B.index[0] + (B.index[-1] - B.index[0]) / 2
eq, t = z01.run2(B, F, 20, 10, 4, 99, 120)
f = z01.metrics(eq, t); a = z01.metrics(eq, t, None, mid); b = z01.metrics(eq, t, mid, None); s = z01.sc4(eq)
print({k: round(v, 3) for k, v in f.items()}); print('SC', {k: round(v, 3) for k, v in s.items()})
print('IS', round(a['net']), 'PF', round(a['PF'], 2), '| OOS', round(b['net']), 'PF', round(b['PF'], 2), '| yarı sınırı', mid.date())
t['y'] = t.et.dt.year; print(t.groupby(['y', 'side']).pnl.sum().unstack().round(0).assign(n=t.groupby('y').size()).to_string())
# komşular
R = []
for k, s_, H in itertools.product([9, 10, 11], [3, 4, 6], [60, 120, 240]):
    e, tt = z01.run2(B, F, 20, k, s_, 99, H); R.append((k, s_, H, round(e.iloc[-1]), len(tt)))
R = pd.DataFrame(R, columns=['k', 's', 'H', 'net', 'n']); print('komşu pozitif', (R.net > 0).mean() * 100, '% | medyan', R.net.median(), '| min', R.net.min())
# monkey: aynı sayıda rastgele giriş (aynı yön dağılımı), aynı çıkış
rng = np.random.default_rng(7); nets = []
for _ in range(200):
    L = np.zeros(len(B), bool); S = np.zeros(len(B), bool)
    idx = rng.choice(np.arange(500, len(B) - 200), size=len(t), replace=False); sd = rng.permutation(t.side.values)
    L[idx[sd > 0]] = True; S[idx[sd < 0]] = True
    e, _ = z01.sim2(F['o'], F['h'], F['l'], F['c'], F['atr'], L, S, 4, 99, 120, np.ones(len(B))); nets.append(e[-1])
print('monkey net P50/P95/P99', np.percentile(nets, [50, 95, 99]).round(0), 'p=', (np.array(nets) >= f['net']).mean())
# grafik
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
m30 = eq.resample('30min').last().dropna(); dd = m30.cummax() - m30
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
ax[0].plot(m30.index, m30.values, '#1f6feb', lw=1); ax[0].set_title('CYBORG | ZEC 2M V0.1 — 30 dk MTM equity (USDT, sabit 500, araştırma motoru)')
ax[0].axvline(mid, color='#888', ls='--', lw=.8); ax[1].fill_between(dd.index, -dd.values, color='#d1242f', lw=0); ax[1].set_ylabel('MTM DD $')
for x in ax: x.grid(alpha=.3)
plt.tight_layout(); plt.savefig('/home/claude/pine_hello_world/ZEC/research/V0.1_equity_dd.png', dpi=110)
json.dump(dict(full=f, IS=a, OOS=b, SC=s), open('/home/claude/pine_hello_world/ZEC/research/V0.1_metrics.json', 'w'), indent=1, default=float)
t.to_csv('/home/claude/pine_hello_world/ZEC/research/V0.1_trades_engine.csv', index=False)
