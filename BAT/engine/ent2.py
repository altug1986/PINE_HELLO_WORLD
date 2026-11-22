import numpy as np, pandas as pd, itertools
from ent import *
df = pd.read_pickle('ent_IS.pkl')
# OI katkısı: aynı H,a,d,hold,sp için fiyat-tek ile kıyas (PF)
base = df[df.fam == 'FİYAT'].set_index(['H', 'a', 'd', 'hold', 'sp'])[['pf', 'net', 'n']]
for fam in ['OI-KURULUM', 'OI-BOŞALMA', 'HACİM-KLİMAKS', 'OI-KALABALIK']:
    g = df[(df.fam == fam) & (df.d == -1) & (df.n >= 100)].join(base, on=['H', 'a', 'd', 'hold', 'sp'], rsuffix='_F')
    print(f'{fam}: PF fiyat-tekten yüksek oran {(g.pf > g.pf_F).mean():.2f} | medyan PF {g.pf.median():.3f} vs {g.pf_F.median():.3f} | medyan işlem {g.n.median():.0f} vs {g.n_F.median():.0f}')
# plato skoru: komşu (tek parametre ±1 adım) medyan net
GRD = dict(H=HS, hold=None, a=None, b=None)
def nb_med(g):
    idx = {(r.H, r.a, r.b, r.hold, r.sp): r.net for r in g.itertuples()}; out = []
    for r in g.itertuples():
        nb = []
        A = AP[r.H] if r.fam != 'BTC-GÖRELİ' else RL[r.H]; Bv = {'OI-KURULUM': BO, 'OI-BOŞALMA': BO, 'HACİM-KLİMAKS': VR}.get(r.fam)
        Bl = Bv[r.H] if Bv else (CR if r.fam == 'OI-KALABALIK' else [0])
        Hl = [r.H // 2, r.H, 2 * r.H]
        for lst, pos in [(A, 1), (Bl, 2), (Hl, 3)]:
            key = [r.H, r.a, r.b, r.hold, r.sp]; j = lst.index(key[pos])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(lst): key2 = list(key); key2[pos] = lst[jj]; v = idx.get(tuple(key2)); nb.append(v) if v is not None else None
        out.append(np.median(nb) if nb else np.nan)
    return out
D0 = load(5); DI = cut(D0, SPLIT); FI = feats(DI); F0 = feats(D0); RG = regimes_bat(DI, FI)
rows = []
for (fam, d), g in df[df.d == -1].groupby(['fam', 'd']):
    g = g.copy(); g['nmed'] = nb_med(g); g['sc'] = 0.5 * g.net + 0.5 * g.nmed
    for _, r in g[g.n >= 200].sort_values('sc', ascending=False).head(2).iterrows():
        gg = (r.fam, int(r.H), r.a, r.b, int(r.d), int(r.hold), int(r.sp))
        R = simR(DI, tg_of(DI, FI, gg)); mk = monkey(DI, R, RG, 1000)
        Ro = simR(D0, tg_of(D0, F0, gg)); m = metrics(D0, Ro)
        rows.append(dict(aile=fam, H=gg[1], a=gg[2], b=gg[3], hold=gg[5], sp=gg[6], IS=r.net, n_IS=r.n, pf_IS=r.pf, komsu=round(r.nmed),
                         monkey_max_p=round(max(mk.values()), 3), zayıf=min(mk, key=lambda k: -mk[k]), OOS=m['OOS'], net=m['net'], pf=m['pf'], p98=m['p98'], n=m['n']))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_pickle('ent2.pkl')
pd.set_option('display.width', 250); print(pd.DataFrame(rows).to_string(index=False))
