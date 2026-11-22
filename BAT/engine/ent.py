import numpy as np, numba as nb, pandas as pd, itertools, sys, time, ast
from lib import simR, summ, cut
from eng import load, TICK
from met import SPLIT, regimes, prep_key, terc, monkey, metrics

HS = [48, 96, 288, 864]
AP = {48: [2, 3, 4.5, 6], 96: [3, 4.5, 6, 8], 288: [5, 7.5, 10, 14], 864: [9, 12, 17, 24]}
BO = {48: [2, 3.5, 6], 96: [3, 5.5, 9], 288: [5, 10, 16], 864: [8, 16, 25]}
VR = {48: [1.5, 2.3, 3.5], 96: [1.5, 2.2, 3.2], 288: [1.3, 1.8, 2.5], 864: [1.15, 1.35, 1.6]}
RL = {48: [1.6, 2.5, 3.4, 5], 96: [2.4, 3.6, 4.8, 7], 288: [4, 6, 8, 12], 864: [7, 10, 13, 20]}
CR = [8, 15, 25]

@nb.njit(cache=True)
def g_ent(c, sig, hold, sl):
    L = len(c); tg = np.zeros(L, np.int64); v = 0; e = 0.0; vb = 0
    for i in range(L):
        if v != 0:
            if (c[i] / e - 1) * v <= -sl / 100 or i - vb >= hold: v = 0
        s = sig[i]
        if s != 0:
            if v != s: v = s; e = c[i]; vb = i
            else: vb = i
        tg[i] = v
    return tg

def feats(D, lag=0):
    c = D['c']; o = np.full(len(c), np.nan); k = 1 + lag; o[k:] = D['oi'][:-k]; v = pd.Series(D['v']); x = D['xc']
    F = {}
    for H in HS:
        sh = lambda a: np.r_[np.full(H, np.nan), a[:-H]]
        F[H] = dict(dP=(c / sh(c) - 1) * 100, dO=(o / sh(o) - 1) * 100, dB=(x / sh(x) - 1) * 100,
                    vr=(v.rolling(H).sum() / (v.rolling(1728).sum() / 1728 * H)).to_numpy())
    F['oiL'] = (pd.Series(o) / pd.Series(o).rolling(1728).mean() - 1).to_numpy() * 100
    F['m4h'] = ((D['t'] + 300000) % 14400000 == 0)  # 4H mum kapanışı
    return F

def sig_of(F, fam, H, a, b, d):
    f = F[H]; dP = f['dP']; s = np.sign(dP)
    if fam == 'FİYAT': cond = np.abs(dP) >= a
    elif fam == 'OI-KURULUM': cond = (np.abs(dP) >= a) & (f['dO'] >= b)
    elif fam == 'OI-BOŞALMA': cond = (np.abs(dP) >= a) & (f['dO'] <= -b)
    elif fam == 'HACİM-KLİMAKS': cond = (np.abs(dP) >= a) & (f['vr'] >= b)
    elif fam == 'OI-KALABALIK': cond = (np.abs(dP) >= a) & (F['oiL'] >= b)
    elif fam == 'BTC-GÖRELİ':
        r = dP - f['dB']; s = np.sign(r); cond = np.abs(r) >= a
    out = np.where(cond & ~np.isnan(dP), s * d, 0); return np.nan_to_num(out).astype(np.int64)

def grid():
    for H in HS:
        for d, hold, sp in itertools.product([1, -1], [H // 2, H, 2 * H], [0, 1]):
            for a in AP[H]:
                yield ('FİYAT', H, a, 0, d, hold, sp)
                for b in BO[H]:
                    yield ('OI-KURULUM', H, a, b, d, hold, sp); yield ('OI-BOŞALMA', H, a, b, d, hold, sp)
                for b in VR[H]: yield ('HACİM-KLİMAKS', H, a, b, d, hold, sp)
                for b in CR: yield ('OI-KALABALIK', H, a, b, d, hold, sp)
            for a in RL[H]: yield ('BTC-GÖRELİ', H, a, 0, d, hold, sp)

def tg_of(D, F, g, sl=10):
    fam, H, a, b, d, hold, sp = g; s = sig_of(F, fam, H, a, b, d)
    if sp: s = np.where(F['m4h'], s, 0)
    return g_ent(D['c'], s, hold, sl)

def regimes_bat(D, F, warm=2000):
    RG = regimes(D, warm); ok = np.arange(len(D['c'])) >= warm
    oiL = F['oiL']; vr = F[288]['vr']
    z0 = pd.Series((D['v'] == 0).astype(float)).rolling(288, min_periods=1).mean().to_numpy()
    RG['OI-SEVİYE'] = prep_key(terc(oiL, ok & ~np.isnan(oiL)))
    RG['HACİM'] = prep_key(terc(vr, ok & ~np.isnan(vr)))
    RG['LİKİDİTE'] = prep_key(terc(z0 + np.random.default_rng(0).random(len(z0)) * 1e-9, ok))
    return RG

if __name__ == '__main__':
    D = cut(load(5), SPLIT); F = feats(D); rows = []; t0 = time.time()
    for g in grid():
        s = summ(simR(D, tg_of(D, F, g))); s.update(fam=g[0], H=g[1], a=g[2], b=g[3], d=g[4], hold=g[5], sp=g[6]); rows.append(s)
    df = pd.DataFrame(rows); df.to_pickle('ent_IS.pkl'); print(len(df), f'{time.time()-t0:.0f}s')
    print(df.groupby(['fam', 'd']).apply(lambda g: pd.Series(dict(ayar=len(g), poz=(g.net > 0).mean().round(2), pf115=((g.pf > 1.15) & (g.n >= 150)).sum(), top=g[g.n >= 150].net.max())), include_groups=False).to_string())
    print(df.groupby(['H', 'sp']).apply(lambda g: (g.net > 0).mean().round(2), include_groups=False).unstack().to_string())
