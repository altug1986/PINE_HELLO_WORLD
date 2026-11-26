exec(open('/home/claude/zecwork/maske2.py').read().split("rows = []")[0])
R = pd.read_pickle('/home/claude/zecwork/maske2.pkl'); pd.set_option('display.width', 250)
print(R[(R.v == '4 s SuperTrend 10/3')][['ADX', 'stop', 'yön', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar']].round(2).to_string(index=False))
S, A = SYS['4 s SuperTrend 10/3']; e = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, S, A, 5, 25.0, 0.05)
for lab, ee in (('V3.3', e0), ('V3.3 + 4s SuperTrend ek', e)):
    x = pd.Series(ee[::30], ixh[::30]); dd = x.cummax() - x; t = dd.idxmax(); pk = x[:t].idxmax(); print(f'{lab}: MaxDD {dd.max():.1f} zirve {pk:%Y-%m-%d %H:%M} → dip {t:%Y-%m-%d %H:%M}')
x = pd.Series(e[::30] - e0[::30], ixh[::30]); print('ek sistemin katkısı yıllık:', x.resample('YE').last().diff().fillna(x.resample('YE').last()).round(0).to_dict())
t0, t1 = pd.Timestamp('2025-09-01'), pd.Timestamp('2026-01-31'); print('ek sistem katkısı aylık (2025-09 → 2026-01):', x[t0:t1].resample('ME').last().diff().round(0).to_dict())
