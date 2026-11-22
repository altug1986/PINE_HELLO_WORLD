import time, numpy as np, pandas as pd
from p1m import build, run, report, trades
from tvcmp63 import tvx
t0 = time.time(); D5 = build(5); print('5 dk veri', len(D5['c']), f'{time.time()-t0:.0f}s', flush=True)
for lab, kw, f in [('port V6.3 · 5 dk · TV benzeri balina', dict(useST=0, whale='tv'), 'tv/v63_tv.xlsx'), ('port V6.6 · 5 dk · TV benzeri balina', dict(useST=1, whale='tv'), None)]:
    out = run(D5, **kw); o = report(D5, out, lab); print(o, flush=True)
    if f:
        R, _ = trades(D5, out); t = D5['t']; P = pd.DataFrame(dict(et=t[R['eb']], xt=t[R['xb']], dr=R['dir'], pnl=R['pnl']))
        T = tvx(f); T = T[T.xt <= t[-1]]; kt = set(zip(T.et, T.dr)); kp = set(zip(P.et, P.dr)); both = kt & kp
        j = T.merge(P, on=['et', 'dr'], suffixes=('_tv', '_p')); same = (j.xt_tv == j.xt_p).mean()
        print(f'  TV {len(T)} işlem {T.pnl.sum():.0f}$ | port {len(P)} {P.pnl.sum():.0f}$ | giriş eşleşmesi TV %{len(both)/len(kt)*100:.1f} port %{len(both)/len(kp)*100:.1f} | aynı girişte aynı çıkış %{same*100:.1f}')
        q = pd.to_datetime(T.et, unit='ms').dt.year; T['m'] = [k in kp for k in zip(T.et, T.dr)]; print('  yıl eşleşme %', (T.groupby(q).m.mean() * 100).round(1).to_dict())
