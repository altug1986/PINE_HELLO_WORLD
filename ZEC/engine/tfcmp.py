import sys, time; sys.path.insert(0, '/home/claude/pine_hello_world/ZEC/engine')
import numpy as np, pandas as pd, ztf, z01
rows = []
for tf in (2, 1, 3, 5):
    t0 = time.time(); D = ztf.build(tf); e, t = ztf.run(D)
    mid = D['B'].index[D['mid']]; f = z01.metrics(e, t); A = z01.metrics(e, t, None, mid); Bm = z01.metrics(e, t, mid, None); sc = z01.sc4(e); rd = ztf.ride(D, t)
    y = t.groupby(t.et.dt.year).pnl.sum().round(0).astype(int).tolist()
    rows.append(dict(TF=f'{tf} dk', kural=len(D['rules']), n=f['n'], net=f['net'], IS=A['net'], OOS=Bm['net'], PF=f['PF'], MaxDD=f['MaxDD'], P99=f['P99'], Calmar=f['Calmar'],
                     Sharpe=f['Sharpe'], SC4=sc['SC4'], s10=f['s10'], s1h=f['s1h'], maeMax=f['maeMax'], short=t[t.side < 0].pnl.sum(),
                     P2_IS=rd['IS'][0], P2_OOS=rd['OOS'][0], doğru_IS=rd['IS'][1], doğru_OOS=rd['OOS'][1], sürüş_IS=rd['IS'][2], sürüş_OOS=rd['OOS'][2], yıllar=y))
    print(tf, 'dk tamam', round(time.time() - t0), 's | kurallar:', ', '.join(D['rules']), flush=True)
    del D
R = pd.DataFrame(rows); R.to_csv('/home/claude/pine_hello_world/ZEC/research/TF_REGULASYON.csv', index=False)
pd.set_option('display.width', 260); print(R.round(2).to_string(index=False))
