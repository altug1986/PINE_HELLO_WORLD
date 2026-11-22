import numpy as np, pandas as pd, itertools, json, numba as nb
from f2 import *
from met import metrics, regimes, monkey, SPLIT
K = ['net', 'pct', 'n', 'pf', 'win', 'sharpe', 'sortino', 'calmar', 'p95', 'p98', 'p99', 'mddc', 'mddi', 'ulcer', 'upi', 'IS', 'OOS', 'posM', 'last12', 'fee', 'yr']
D = load(5); B = json.load(open('f2_best.json')); order = B['order']; prm = B['prm']
def full(D, order, prm, lag=0, blag=0):
    R = simR(D, chain([motor(D, m, prm[m], lag, blag) for m in order])); return metrics(D, R), R
out = {}
# 1. BAZ vs YENİ
m_new, R = full(D, order, prm)
m_b0 = metrics(D, go(D, P(), FULL)); m_col = metrics(D, go(D, P(), np.array([0,0,1,0,0,0,0,0,0], np.int64)))
T = pd.DataFrame([m_b0, m_col, m_new], index=['BAZ-0 (PEPE V11.7 ayarı)', 'Sade baz (çarpışma)', 'YENİ: LİK>Z>RSI'])[K]
print(T.T.to_string(), flush=True); T.to_pickle('f3_tab.pkl')
# 2. 6 aylık bloklar
xt = pd.to_datetime(D['t'][R['xb']], unit='ms'); p = pd.Series(R['pnl'], index=xt)
blk = p.groupby([xt.year, (xt.month - 1) // 6]).agg(['sum', 'count']); blk['pf'] = p.groupby([xt.year, (xt.month - 1) // 6]).apply(lambda x: x[x > 0].sum() / max(-x[x < 0].sum(), 1e-9))
print(blk.round(2).T.to_string()); print('pozitif blok', (blk['sum'] > 0).sum(), '/', len(blk), flush=True)
# 3. OI / BTC gecikme: LİK katkısı her gecikmede
for lag in [0, 1, 2]:
    a, _ = full(D, order, prm, lag); b, _ = full(D, [m for m in order if m != 'LIQ'], prm)
    print(f'OI gecikme {lag}: zincir {a["net"]} PF {a["pf"]} P98 {a["p98"]} | LİK yok {b["net"]} PF {b["pf"]} P98 {b["p98"]} | IS {a["IS"]}/{b["IS"]} OOS {a["OOS"]}/{b["OOS"]}', flush=True)
# 4. Plato: her parametre ±1 adım
rows = []
for mm in order:
    for k in GR[mm]:
        vals = GR[mm][k]; j = vals.index(prm[mm][k])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                q = json.loads(json.dumps(prm)); q[mm][k] = vals[jj]; a, _ = full(D, order, q)
                rows.append(dict(p=f'{mm}.{k}={vals[jj]}', net=a['net'], IS=a['IS'], OOS=a['OOS'], pf=a['pf'], p98=a['p98']))
pl = pd.DataFrame(rows); print(pl.to_string(index=False))
print('plato: net>=%90 baz', (pl.net >= 0.9 * m_new['net']).mean().round(2), '| OOS>0', (pl.OOS > 0).mean().round(2), flush=True)
# 5. Monkey (tam dönem) + MC
RG = regimes(D); mk = monkey(D, R, RG, 2000); print('MONKEY p', {k: round(v, 4) for k, v in mk.items()})
pn = R['pnl']; rng = np.random.default_rng(1); N = 10000
def dd(x):
    cp = np.cumsum(x); return (np.maximum.accumulate(np.r_[0, cp])[1:] - cp).max()
bs = np.array([rng.choice(pn, len(pn)) for _ in range(N)]); bnet = bs.sum(1); bdd = np.array([dd(x) for x in bs])
sh = np.array([dd(rng.permutation(pn)) for _ in range(N)])
skip = np.array([dd(np.where(rng.random(len(pn)) < .05, 0, pn - .05)) for _ in range(2000)])
print(f'MC bootstrap net P1/P5/P50 {np.percentile(bnet,[1,5,50]).round()} zarar olasılığı {(bnet<0).mean():.4f} | DD P50/P95/P99 {np.percentile(bdd,[50,95,99]).round()}')
print(f'MC sıra karıştırma DD P50/P95/P99 {np.percentile(sh,[50,95,99]).round()} | gerçek {dd(pn):.0f} | %5 atla+0.05$ DD P95 {np.percentile(skip,95):.0f}')
# 6. Stres %0.08 + 3 tick
q = R['q']; st = pn - 0.0004 * q * (R['ep'] + R['xp']) - 2 * TICK * q * 2
print(f'STRES net {st.sum():.0f} PF {st[st>0].sum()/-st[st<0].sum():.3f} | IS {st[D["t"][R["xb"]]<SPLIT].sum():.0f} OOS {st[D["t"][R["xb"]]>=SPLIT].sum():.0f}')
# motor katkısı (tam dönem, hangi motor hedefi belirledi)
tgs = {mm: motor(D, mm, prm[mm]) for mm in order}; src = np.zeros(len(D['c']), '<U3')
for mm in reversed(order): src = np.where(tgs[mm] != 0, mm, src)
s = pd.Series(R['pnl']).groupby(src[R['eb']]).agg(['sum', 'count']); print('motor katkısı (giriş motoru)', s.to_dict())
np.save('f3_R.npy', np.c_[R['eb'], R['xb'], R['dir'], R['ep'], R['xp'], R['q'], R['pnl']])
