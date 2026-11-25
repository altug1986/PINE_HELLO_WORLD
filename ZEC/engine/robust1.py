# Gama: R-ailesi + rejimde taşı + çoklu motor → OOS, 6 ay blokları, MC bootstrap/shuffle, 12 ay ileri DD (blok bootstrap), monkey yön/zaman, stres
exec(open('/home/claude/zecwork/erhan1.py').read().split("C = {")[0])
import functools, ast; rng = np.random.default_rng(7)
select = functools.lru_cache(None)(select)
B0 = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/BOB_1DK_2HAFTA.csv'); H0 = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/HOLD_1DK.csv')
g = B0[(B0.IS > 0) & (B0.OOS > 0) & (B0.zayıf > 0) & B0.n.between(1000, 2500)].sort_values('BoB', ascending=False).head(6)
G = {f'R-{i+1}': (r.thr, r.hz, r.kL, r.kS, 0, 3, r.ts, r.tH) for i, (_, r) in enumerate(g.iterrows())}
G['R5'] = (15, 240, 2, 3, 0, 3, 16.0, 1920)
hh = H0[(H0.IS > 0) & (H0.OOS > 0) & (H0.zayıf > 0) & (H0['mod'] > 0)]; hh = hh.assign(sk=hh.net / hh.DDp98 * hh.R2 * hh.blokPoz).sort_values('sk', ascending=False).head(4)
for i, (_, r) in enumerate(hh.iterrows()): G[f'TAŞI-{i+1}'] = (r.thr, r.hz, r.kL, r.kS, r['mod'], r.k4, r.ts, r.tH)
G['TAŞI-maks'] = (15, 240, 2, 4, 1, 8, 16.0, 1920); G['TAŞI-dengeli'] = (10, 480, 2, 3, 2, 3, 24.0, 1920)
G = {k: tuple(float(x) for x in v) for k, v in G.items()}; uniq = {}
for k, v in G.items(): uniq.setdefault(v, k)
G = {k: v for v, k in uniq.items()}
ens = ast.literal_eval('[' + pd.read_csv('/home/claude/pine_hello_world/ZEC/research/ENS_1DK.csv').motorlar.iloc[-1].replace(' | ', ',').replace('np.float64(', '').replace('), ', ', ') + ']')
yrs = n / 525600; b6 = np.searchsorted(idx.values, pd.date_range('2021-01-01', idx[-1], freq='6MS').values); b6 = b6[b6 > 0]
def run1(cfg):
    thr, hz, kL, kS, mod, k4, ts, tH = cfg; L, S = select(int(thr), int(hz)); aL = (L >= kL); aS = (S >= kS) & ~aL
    eq, T = z03.sim10(o, h, l, c, a, aL, aS, float(ts), int(tH), spk, 60, RGi, A4 * k4, int(mod)); return eq, np.c_[T, np.ones(len(T))]
def mdd(x, ax=-1): return (np.maximum.accumulate(x, axis=ax) - x).max(axis=ax)
out = []; EQS = {}
for nm, cfg in list(G.items()) + [('ÇOKLU-6', None)]:
    if cfg is None:
        R_ = [run1(tuple(map(float, e))) for e in ens]; eq = sum(r[0] for r in R_) / len(ens); T = np.concatenate([r[1] for r in R_]); T[:, 5] /= len(ens); T[:, 8] /= len(ens)
    else: eq, T = run1(cfg)
    T = T[np.argsort(T[:, 1])]; w = T[:, 8]; tp = T[:, 5]; e_, x_, d_ = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2]; q = 500 * w / T[:, 3]
    x30 = eq[::30]; d30 = np.maximum.accumulate(x30) - x30; EQS[nm] = x30
    r = dict(sistem=nm, n=len(T), net=eq[-1], IS=eq[mid] - eq[0], OOS=eq[-1] - eq[mid], MaxDD=d30.max(), DDp98=np.percentile(d30, 98), Calmar=eq[-1] / yrs / d30.max())
    bp = np.diff(np.r_[0, eq[b6], eq[-1]]); r['6ayPoz'] = f'{(bp > 0).sum()}/{len(bp)}'; r['6ayPoz%'] = (bp > 0).mean() * 100
    # MC bootstrap (yerine koymalı) + %5 işlem atla + işlem başı +0,05 $; shuffle → kapanmış işlem dizisi DD
    I = rng.integers(0, len(tp), (5000, len(tp))); keep = rng.random((5000, len(tp))) > 0.05
    P = np.where(keep, tp[I] - 0.05, 0.0); cs = np.cumsum(P, 1); r['MC_netP1'] = np.percentile(cs[:, -1], 1); r['MC_netP5'] = np.percentile(cs[:, -1], 5); r['MC_zarar%'] = (cs[:, -1] < 0).mean() * 100
    dd = mdd(np.c_[np.zeros(5000), cs]); r['MC_DDp95'] = np.percentile(dd, 95); r['MC_DDp99'] = np.percentile(dd, 99)
    S_ = np.cumsum(np.array([rng.permutation(tp) for _ in range(2000)]), 1); dds = mdd(np.c_[np.zeros(2000), S_]); r['SHF_DDp95'] = np.percentile(dds, 95); r['SHF_DDp99'] = np.percentile(dds, 99)
    # 12 ay ileri MTM DD: 30 dk artışlar, 1 günlük bloklar (48 adım) blok bootstrap, 2.000 yol
    inc = np.diff(x30); nb = len(inc) // 48; Bk = inc[:nb * 48].reshape(nb, 48); path = np.cumsum(Bk[rng.integers(0, nb, (2000, 365))].reshape(2000, -1), 1)
    fd = mdd(np.c_[np.zeros(2000), path]); fn = path[:, -1]
    for p in (90, 95, 99): r[f'12ay_DDp{p}'] = np.percentile(fd, p)
    r['12ay_netP5'] = np.percentile(fn, 5); r['12ay_zarar%'] = (fn < 0).mean() * 100
    # Monkey (kapanıştan kapanışa; sistemin aynı ölçüyle neti)
    fee = lambda ep, xp, qq: 0.0004 * qq * (ep + xp) + 2 * 0.01 * qq
    sysCC = (q * (c[x_] - c[e_]) * d_ - fee(c[e_], c[x_], q)).sum()
    D = rng.choice([-1.0, 1.0], (1000, len(tp))); mY = (q * (c[x_] - c[e_]) * D - fee(c[e_], c[x_], q)).sum(1)
    du = x_ - e_; E2 = (rng.random((1000, len(tp))) * (n - 1 - du)).astype(np.int64); X2 = E2 + du
    qZ = 500 * w / c[E2]; mZ = (qZ * (c[X2] - c[E2]) * d_ - fee(c[E2], c[X2], qZ)).sum(1)
    r.update(sysCC=sysCC, monkeyYönMaks=mY.max(), monkeyYön_p=(mY >= sysCC).mean(), monkeyZamanMaks=mZ.max(), monkeyZaman_p=(mZ >= sysCC).mean())
    q2 = 500 * w / T[:, 3]; xc = 0.0004 * q2 * (T[:, 3] + T[:, 4]) + 4 * 0.01 * q2; cum = np.zeros(n); np.add.at(cum, x_, xc); r['stresNet'] = eq[-1] - cum.sum()
    r['yatayGün'] = flat(x30)[0]; r['top3%'] = np.sort(tp)[-3:].sum() / eq[-1] * 100
    for k_, (a_, b_) in WIN.items(): r[k_] = eq[b_] - eq[a_]
    r['ayar'] = str(cfg); out.append(r); pass
R = pd.DataFrame(out); R.to_csv('/home/claude/pine_hello_world/ZEC/research/ROBUST_GAMA_1DK.csv', index=False)
# Gama düzeyinde OOS: IS'te en iyiyi seç → OOS'taki sırası (ve tersi)
def bob(x, a_, b_):
    x = x[a_:b_]; dd = np.maximum.accumulate(x) - x; return (x[-1] - x[0]) / max(np.percentile(dd, 98), 1) * np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2
m30 = mid // 30; isS = {k: bob(v, 0, m30) for k, v in EQS.items()}; ooS = {k: bob(v, m30, None) for k, v in EQS.items()}
bi = max(isS, key=isS.get); bo = max(ooS, key=ooS.get); rk = lambda d, k: sorted(d, key=d.get, reverse=True).index(k) + 1
print(f'\nGama OOS: IS en iyisi {bi} → OOS sırası {rk(ooS, bi)}/{len(EQS)} · OOS en iyisi {bo} → IS sırası {rk(isS, bo)}/{len(EQS)}')
pd.set_option('display.width', 260); print(R.drop(columns='ayar').set_index('sistem').T.round(2).to_string()); print(R[['sistem', 'ayar']].to_string(index=False))
