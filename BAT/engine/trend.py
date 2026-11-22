import numpy as np, pandas as pd, itertools, time
from eng import load, dmi, rma, trr, TICK
from lib import simR
from met import metrics, SPLIT
D = load(5); c5 = D['c']; t5 = D['t']; n = len(c5)
def htf(P, src='bat'):
    k = t5 // P
    df = pd.DataFrame(dict(k=k, o=D['o'], h=D['h'], l=D['l'], c=c5)) if src == 'bat' else pd.DataFrame(dict(k=k, c=D['xc']))
    g = df.groupby('k'); H = pd.DataFrame(dict(c=g.c.last())) if src != 'bat' else pd.DataFrame(dict(o=g.o.first(), h=g.h.max(), l=g.l.min(), c=g.c.last()))
    return H, k
def mapv(vals, H, k):  # lookahead_on + [1]: önceki tamamlanmış HTF mumu
    s = pd.Series(np.r_[np.nan, vals[:-1]], index=H.index); return s.reindex(k).to_numpy()
def hold_sig(sig):  # 0 = önceki yönü koru
    s = pd.Series(np.where(sig == 0, np.nan, sig)).ffill().fillna(0).to_numpy(); return s.astype(np.int64)
TF = {'4S': 14400000, '1G': 86400000}
HT = {k: htf(P) for k, P in TF.items()}; HX = {k: htf(P, 'btc') for k, P in TF.items()}
fams = []
for tf, N, x, md in itertools.product(TF, [3, 5, 10, 20, 30, 60], [0, 2, 5, 10], ['flat', 'koru']):
    def f(tf=tf, N=N, x=x, md=md):
        H, k = HT[tf]; m = (H.c / H.c.shift(N) - 1).to_numpy() * 100; s = np.where(m >= x, 1, np.where(m <= -x, -1, 0)); s = np.nan_to_num(mapv(s, H, k))
        return hold_sig(s) if md == 'koru' else s.astype(np.int64)
    fams.append(('MOMENTUM', f'{tf} N{N} x{x} {md}', f))
for tf, N, ex in itertools.product(TF, [5, 10, 20, 55, 100], [0, 2]):
    def f(tf=tf, N=N, ex=ex):
        H, k = HT[tf]; hh = H.h.rolling(N).max().shift(1); ll = H.l.rolling(N).min().shift(1)
        e = max(N // 2, 2); xh = H.h.rolling(e).max().shift(1); xl = H.l.rolling(e).min().shift(1); s = np.zeros(len(H)); v = 0
        for i in range(len(H)):
            cc = H.c.iat[i]
            if ex == 2 and v == 1 and cc < xl.iat[i]: v = 0
            if ex == 2 and v == -1 and cc > xh.iat[i]: v = 0
            if cc > hh.iat[i]: v = 1
            elif cc < ll.iat[i]: v = -1
            s[i] = v
        return np.nan_to_num(mapv(s, H, k)).astype(np.int64)
    fams.append(('DONCHIAN', f'{tf} N{N} {"turtle" if ex else "dönüş"}', f))
for tf, (fa, sl) in itertools.product(TF, [(5, 20), (10, 30), (20, 50), (20, 100), (50, 200), (9, 21)]):
    def f(tf=tf, fa=fa, sl=sl):
        H, k = HT[tf]; s = np.sign(H.c.ewm(span=fa, adjust=False).mean() - H.c.ewm(span=sl, adjust=False).mean()).to_numpy()
        return np.nan_to_num(mapv(s, H, k)).astype(np.int64)
    fams.append(('EMA', f'{tf} {fa}/{sl}', f))
for tf, al, mu in itertools.product(TF, [10, 14], [2, 3, 4]):
    def f(tf=tf, al=al, mu=mu):
        H, k = HT[tf]; h, l, c = H.h.to_numpy(), H.l.to_numpy(), H.c.to_numpy(); a = rma(trr(h, l, c, True), al); hl = (h + l) / 2
        up = hl - mu * a; dn = hl + mu * a; s = np.zeros(len(c)); v = 1; U = up[0]; Dn = dn[0]
        for i in range(1, len(c)):
            if np.isnan(a[i]): continue
            U = max(up[i], U) if c[i - 1] > U else up[i]; Dn = min(dn[i], Dn) if c[i - 1] < Dn else dn[i]
            if v == -1 and c[i] > Dn: v = 1
            elif v == 1 and c[i] < U: v = -1
            s[i] = v
        return np.nan_to_num(mapv(s, H, k)).astype(np.int64)
    fams.append(('SUPERTREND', f'{tf} atr{al} x{mu}', f))
t = pd.to_datetime(t5, unit='ms'); blk = ((t.year - 2020) * 2 + (t.month - 1) // 6).to_numpy(); NB = blk.max() + 1
rows = []; T0 = time.time()
for fam, prm, g in fams:
    tg = g(); R = simR(D, tg); p = R['pnl']; xt = t5[R['xb']]
    rows.append(dict(fam=fam, prm=prm, net=round(p.sum()), n=len(p), pf=round(p[p > 0].sum() / max(-p[p < 0].sum(), 1e-9), 2),
                     IS=round(p[xt < SPLIT].sum()), OOS=round(p[xt >= SPLIT].sum()), pb=np.bincount(blk[R['xb']], weights=p, minlength=NB), tg=tg))
df = pd.DataFrame(rows); print(len(df), f'{time.time()-T0:.0f}s')
out = []
for fam, g in df.groupby('fam', sort=False):
    PB = np.stack(g.pb.values); wf = []
    for b in range(2, NB):
        j = PB[:, :b].sum(1).argmax(); wf.append(PB[j, b])
    wf = np.array(wf); bi = g.IS.idxmax()
    out.append(dict(aile=fam, ayar=len(g), tam_poz=round((g.net > 0).mean(), 2), IS_poz=round((g.IS > 0).mean(), 2), OOS_poz=round((g.OOS > 0).mean(), 2),
                    YP_net=round(wf.sum()), YP_poz=f'{(wf > 0).sum()}/{len(wf)}', IS_secim=f"{df.prm[bi]} IS {df.IS[bi]} OOS {df.OOS[bi]} n {df.n[bi]} PF {df.pf[bi]}"))
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 80); print(pd.DataFrame(out).to_string(index=False))
print(df.sort_values('net', ascending=False).head(8)[['fam', 'prm', 'net', 'n', 'pf', 'IS', 'OOS']].to_string(index=False))
df.drop(columns=['tg']).to_pickle('trend.pkl')
