import numpy as np, numba as nb, pandas as pd, json, time, sys, itertools
from scan3 import D, F, system, simR, metrics, SPLIT, TICK, comp
from eng import dmi, htf_map
from ent import g_ent
from vote import vote_pos
# V5.3A + (1) KRİTİK ANDA BTC 1S ADX İLE ERKEN TERS YÖN  (2) BTC/BAT 30 DK HACİM ORANI METRİĞİ
d = json.load(open('v53_11545.json')); CFG = d['cfg']; CFG['p'] = {int(k): v for k, v in CFG['p'].items()}; S0, RA0 = d['S'], d['rA']
c = D['c']; t = D['t']; n = len(c); z = dict(np.load('d5.npz'))
x = z['x60']; pdi, mdi, ax = dmi(x[1], x[2], x[3], 14, 14)
BADX = np.nan_to_num(htf_map(ax, z['x60_k'], z['x60_map'])); BDI = np.nan_to_num(htf_map(pdi - mdi, z['x60_k'], z['x60_map']))
b = pd.read_csv('btc1m.csv.gz', usecols=['time', 'close', 'volume']); k5 = (b.time.to_numpy() // 300000).astype(np.int64)
bq = pd.Series(b.volume.to_numpy() * b.close.to_numpy()).groupby(k5).sum(); BQ = bq.reindex(t // 300000).fillna(0).to_numpy()
AQ = D['v'] * c
r30 = pd.Series(AQ).rolling(6).sum() / pd.Series(BQ).rolling(6).sum().replace(0, np.nan)
NR = np.nan_to_num((r30 / r30.rolling(1728).mean()).to_numpy(), nan=1.0)       # BAT/BTC 30 dk hacim oranı ÷ 6 günlük ortalaması

@nb.njit(cache=True)
def layer(c, tg, BADX, BDI, NR, S, rA, L, a, vmode, vk, cS, N, tick):
    n = len(c); out = np.zeros(n, np.int64); sh = 0; ep = 0.0; lock = 0; lext = 0.0; prevT = 0; cnt = 0; cPx = 0.0; cBar = 0; nrev = 0
    for i in range(n):
        t_ = tg[i]
        if t_ != prevT:
            lock = 0; cnt = 0; prevT = t_
        want = t_
        if cnt != 0:
            g = (c[i] / cPx - 1) * cnt * 100
            if g <= -cS or i - cBar >= N: cnt = 0
            else: want = cnt
        if cnt == 0:
            if sh != 0 and want == sh and L < 50:
                loss = (c[i] / ep - 1) * sh * 100
                ok = vmode == 0 or (vmode == 1 and NR[i] >= vk) or (vmode == 2 and NR[i] <= vk)
                if loss <= -L and BADX[i] > a and BDI[i] * sh < 0 and ok:
                    cnt = -sh; cPx = c[i]; cBar = i; want = cnt; nrev += 1
            if cnt == 0:
                if sh != 0 and want == sh and (c[i] / ep - 1) * sh * 100 <= -S:
                    lock = sh; lext = c[i]; want = 0
                if lock != 0 and want == lock:
                    if lock > 0: lext = min(lext, c[i])
                    else: lext = max(lext, c[i])
                    if (c[i] / lext - 1) * lock * 100 >= rA: lock = 0
                    else: want = 0
        if want != sh:
            sh = want; ep = c[i] + tick * want
        out[i] = sh
    return out, nrev

X0 = np.stack([comp(j, CFG['p'][j]) for j in range(1, 15)])
d288 = np.r_[np.full(288, np.nan), (c[288:] / c[:-288] - 1) * 100]
def comp15(k, a):
    s = np.where((NR >= k) & (np.abs(np.nan_to_num(d288)) >= a), -np.sign(np.nan_to_num(d288)), 0).astype(np.int64)
    return g_ent(c, s, 288, 10.0)
def target(w15=0.0, k15=2.0, a15=4.0):
    if w15 == 0: return vote_pos(X0, np.array(CFG['w'], float), CFG['on'], CFG['off'], 1)[0]
    X = np.vstack([X0, comp15(k15, a15)[None, :]]); return vote_pos(X, np.array(CFG['w'] + [w15], float), CFG['on'], CFG['off'], 1)[0]
def run(p, tg=None):
    tg = target(p.get('w15', 0.0), p.get('k15', 2.0), p.get('a15', 4.0)) if tg is None else tg
    return layer(c, tg, BADX, BDI, NR, S0, RA0, p['L'], p['a'], p['vm'], p['vk'], p['cS'], p['N'], TICK)
if __name__ == '__main__':
    k0 = np.searchsorted(t, SPLIT); km = np.searchsorted(t, np.int64(pd.Timestamp('2022-06-15').value // 10**6))
    def cutD(a_, b_): return {kk: (v[a_:b_] if isinstance(v, np.ndarray) and len(v) == n else v) for kk, v in D.items()}
    H = [cutD(0, km), cutD(km, k0)]
    def halves(tg):
        out = []
        for (a_, b_), Dh in zip([(0, km), (km, k0)], H):
            m = metrics(Dh, simR(Dh, tg[a_:b_])); out.append(m['net'] - 1.5 * m['p98'])
        return min(out), out
    tgB = target(); s0 = halves(run(dict(L=99.0, a=0.0, vm=0, vk=0.0, cS=3.0, N=144), tgB)[0]); print('V5.3A ilk dönem yarı skorları', [round(v) for v in s0[1]], flush=True)
    seed = int(sys.argv[1]); rows = []; t0 = time.time()
    if seed in (0, 1):
        G = list(itertools.product([3.0, 4.0, 5.0, 6.0], [20.0, 25.0, 30.0, 35.0, 40.0], [(0, 0.0), (1, 1.5), (1, 2.0), (2, 1.0)], [2.0, 3.0, 5.0], [72, 144, 288]))[seed::2]
        for L, a, (vm, vk), cS, N in G:
            p = dict(L=L, a=a, vm=vm, vk=vk, cS=cS, N=N); tg, nr = run(p, tgB); s, hs = halves(tg); rows.append(dict(**p, dönüş=nr, sc=s, A=hs[0], B=hs[1]))
    else:
        for w15, k15, a15 in itertools.product([0.5, 1.0, 2.0, 3.0, 4.0], [1.5, 2.0, 3.0], [3.0, 4.0, 5.0]):
            p = dict(L=99.0, a=0.0, vm=0, vk=0.0, cS=3.0, N=144, w15=w15, k15=k15, a15=a15); tg, _ = run(p); s, hs = halves(tg); rows.append(dict(**p, sc=s, A=hs[0], B=hs[1]))
    df = pd.DataFrame(rows); df.to_pickle(f'v54_{seed}.pkl'); print(len(df), 'varyant', f'{time.time()-t0:.0f}s', '| iki yarıda da V5.3A üstü:', int(((df.A > s0[1][0]) & (df.B > s0[1][1])).sum()))
    pd.set_option('display.width', 250); print(df.sort_values('sc', ascending=False).head(5).to_string(index=False))
