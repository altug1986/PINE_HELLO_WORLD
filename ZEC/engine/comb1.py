import numpy as np, pandas as pd, glob
t = pd.concat([pd.read_parquet(f, columns=['time']).time for f in sorted(glob.glob('/home/claude/master/*.parquet'))]).values
i30 = pd.to_datetime(t[::30], unit='ms'); Sp = pd.read_pickle('/home/claude/port/portfoy_10dk_utc.pkl')
E = {'ÇEVİK-1': np.load('eq_cevik_1.npy'), 'ÇEVİK-2': np.load('eq_cevik_2.npy'), 'AKRAN-A': np.load('eq_A.npy'), 'AKRAN-C': np.load('eq_C.npy')}
P = Sp.resample('30min').last().ffill(); a, b = pd.Timestamp('2023-05-08'), pd.Timestamp('2026-11-13 23:30')
P = P[a:b]; P = P - P.iloc[0]; out = []
def m(x): dd = np.maximum.accumulate(x) - x; return dd.max(), np.percentile(dd, 99)
y = (b - a).days / 365; p0 = m(P.values)
out.append(dict(sistem='Portföy 4.600 tek', net=P.iloc[-1], MaxDD=p0[0], DDp99=p0[1], Calmar=P.iloc[-1] / y / p0[0], kor=np.nan, son30_port_penceresi=P.iloc[-1] - P[P.index >= b - pd.Timedelta(days=30)].iloc[0]))
for k, x in E.items():
    z = pd.Series(x, index=i30[:len(x)])[a:b].reindex(P.index).ffill(); z = z - z.iloc[0]
    dz = z.resample('1D').last().diff(); dp = P.resample('1D').last().diff(); kor = dz.corr(dp)
    for nom in (250, 500):
        C_ = P.values + z.values * nom / 500; mm = m(C_)
        out.append(dict(sistem=f'Portföy + {k} ({nom}$)', net=C_[-1], MaxDD=mm[0], DDp99=mm[1], Calmar=C_[-1] / y / mm[0], kor=kor, son30_port_penceresi=(C_[-1] - C_[P.index >= b - pd.Timedelta(days=30)][0])))
pd.set_option('display.width', 200); print(pd.DataFrame(out).round(2).to_string(index=False))
# Portföyün en kötü 10 günü ve ÇEVİK-1 aynı gün
dp = P.resample('1D').last().diff(); z = pd.Series(E['ÇEVİK-1'], index=i30[:len(E['ÇEVİK-1'])]).resample('1D').last().diff()
w = dp.nsmallest(10); print('\nPortföyün en kötü 10 günü → ÇEVİK-1 aynı gün:'); print(pd.DataFrame({'portföy': w, 'ÇEVİK-1': z.reindex(w.index)}).round(1).T.to_string())
