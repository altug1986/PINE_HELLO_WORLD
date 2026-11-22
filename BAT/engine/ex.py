import numpy as np, numba as nb, pandas as pd, itertools, sys
from ent import *

@nb.njit(cache=True)
def g_ex(c, sig, ref, oi, v, vavg, hold, sl, tp, trA, trG, oiDn, oiDnG, vk, vg, retr, oiAd, oiAdL, rearm):
    # sig: yön sinyali; ref: sinyal hareketinin başlangıç fiyatı (c[i-H]); oi: oi[1] serisi
    L = len(c); tg = np.zeros(L, np.int64); v_ = 0; e = 0.0; vb = 0; pk = 0.0; oie = 0.0; tgt = 0.0; lock = 0
    for i in range(L):
        s = sig[i]
        if lock != 0 and s != lock: lock = 0
        if v_ != 0:
            r = (c[i] / e - 1) * v_; pk = max(pk, r); ex = False; early = False
            if r <= -sl / 100 or i - vb >= hold: ex = True
            elif tp > 0 and r >= tp / 100: ex = True; early = True
            elif trA > 0 and pk >= trA / 100 and pk - r >= trG / 100: ex = True; early = True
            elif retr > 0 and (c[i] - tgt) * v_ >= 0: ex = True; early = True
            elif i >= 2 and oiDn > 0 and not np.isnan(oi[i]) and not np.isnan(oi[i - 2]) and (oi[i] / oi[i - 2] - 1) * 100 <= -oiDn and r >= oiDnG / 100: ex = True; early = True
            elif vk > 0 and vavg[i] > 0 and v[i] >= vk * vavg[i] and (c[i] / c[i - 1] - 1) * v_ >= vg / 100 and r > 0: ex = True; early = True
            elif oiAd > 0 and oie > 0 and not np.isnan(oi[i]) and (oi[i] / oie - 1) * 100 >= oiAd and r <= -oiAdL / 100: ex = True; early = True
            if ex:
                if early and rearm == 1: lock = v_
                v_ = 0
        if s != 0 and s != lock:
            if v_ != s:
                v_ = s; e = c[i]; vb = i; pk = 0.0; oie = oi[i] if not np.isnan(oi[i]) else 0.0
                tgt = e + retr * (ref[i] - e)
            else:
                vb = i
        tg[i] = v_
    return tg

XG = dict(tp=[0, 2, 3, 4, 6], trA=[0, 2, 3, 5], trG=[1, 2], oiDn=[0, 1, 2, 3], oiDnG=[0, 1, 2], vk=[0, 3, 5, 8], vg=[0.5, 1],
          retr=[0, 0.3, 0.5, 0.7], oiAd=[0, 3, 5, 10], oiAdL=[1, 3], hold=[144, 288, 576], sl=[5, 8, 10, 15])
XB = dict(tp=0, trA=0, trG=1, oiDn=0, oiDnG=0, vk=0, vg=0.5, retr=0, oiAd=0, oiAdL=1, hold=288, sl=10)

def ex_tg(D, F, ent, x, lag=0, rearm=1):
    fam, H, a, b, d = ent; s = sig_of(F, fam, H, a, b, d); c = D['c']
    ref = np.r_[np.full(H, np.nan), c[:-H]]; ref = np.where(np.isnan(ref), c, ref)
    o = np.full(len(c), np.nan); k = 1 + lag; o[k:] = D['oi'][:-k]
    vav = pd.Series(D['v']).rolling(288).mean().shift(1).fillna(0).to_numpy()
    return g_ex(c, s, ref, o, D['v'], vav, int(x['hold']), x['sl'], x['tp'], x['trA'], x['trG'], x['oiDn'], x['oiDnG'], x['vk'], x['vg'],
                x['retr'], x['oiAd'], x['oiAdL'], rearm)

if __name__ == '__main__':
    D0 = load(5); DI = cut(D0, SPLIT); FI = feats(DI); F0 = feats(D0)
    ENT = [('FİYAT', 288, 5.0, 0, -1), ('OI-KURULUM', 288, 5.0, 5.0, -1)]
    pd.set_option('display.width', 250)
    for ent in ENT:
        b0 = summ(simR(DI, ex_tg(DI, FI, ent, XB))); print('==', ent, 'BAZ çıkış (1 gün tut, %10 stop)', b0)
        rows = []
        # tek tek aile: her çıkış kuralını kendi ızgarasında dene (diğerleri kapalı)
        fams = dict(HEDEF=['tp'], İZ=['trA', 'trG'], GERİDÖNÜŞ=['retr'], OI10_BOŞALMA=['oiDn', 'oiDnG'], HACİM_KLİMAKS=['vk', 'vg'],
                    OI_TERS_KURULUM=['oiAd', 'oiAdL'], SÜRE=['hold'], STOP=['sl'])
        for fn, ks in fams.items():
            for vals in itertools.product(*[XG[k] for k in ks]):
                x = dict(XB); x.update(dict(zip(ks, vals)))
                if x == XB: continue
                s = summ(simR(DI, ex_tg(DI, FI, ent, x))); rows.append(dict(aile=fn, ayar=str(dict(zip(ks, vals))), **s))
        df = pd.DataFrame(rows); df['Δnet'] = df.net - b0['net']; df['Δpf'] = (df.pf - b0['pf']).round(3)
        print(df.groupby('aile').apply(lambda g: pd.Series(dict(ayar=len(g), iyi_oran=((g['Δnet'] > 0) & (g['Δpf'] > 0)).mean().round(2), en_iyi=g.loc[g['Δnet'].idxmax(), 'ayar'], Δnet=g['Δnet'].max(), Δpf=g.loc[g['Δnet'].idxmax(), 'Δpf'])), include_groups=False).to_string())
        df.to_pickle(f'ex_{ent[0]}.pkl')
