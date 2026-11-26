# Üst zaman dilimli EK sistem, her zaman değil: V3.3 FLAT iken ve 4 s ADX ≥ eşik iken üst sistemin yönünde; stop tavanı, yön seçeneği
exec(open('/home/claude/zecwork/maske.py').read().split("F0 = np.zeros(n); e0 = comb(")[0])
F0 = np.zeros(n); e0 = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, F0, F0, 0, 0.0, 0.1); B = met(e0, 'V3.3')
rows = []
for nm in ('4 s SuperTrend 10/3', '4 s EMA10/30', '4 s Donchian 20/10', '4 s EMA20/50', '4 s Donchian 55/20', '1 s EMA50/200'):
    S, A = SYS[nm]
    for athr, cap, side in itertools.product((25.0, 30.0, 35.0), (0.03, 0.05, 0.08, 0.10), ('iki yön', 'sadece long')):
        Sx = np.where(S > 0, 1.0, 0.0) if side == 'sadece long' else S
        e = comb(o_, h_, l_, cc, 0.01, rp, rx, rn, Sx, A, 5, athr, cap); rows.append(met(e, nm, ADX=athr, stop=cap, yön=side))
R = pd.DataFrame(rows)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R['ikiYarı'] = (R.dIS > 0) & (R.dOOS > 0); R['kapı'] = (R.net >= B['net'] * 1.06) & (R.MaxDD <= B['MaxDD'] + 0.01) & (R.Ulcer <= B['Ulcer'] + 0.01) & R.ikiYarı
R.to_pickle('/home/claude/zecwork/maske2.pkl'); pd.set_option('display.width', 300)
print(f"{len(R)} varyant | iki yarı artı {R.ikiYarı.sum()} | kapı {R.kapı.sum()} | MaxDD≤baz ve iki yarı artı: {((R.MaxDD <= B['MaxDD'] + 0.01) & R.ikiYarı).sum()}")
print(R.groupby(['v']).agg(medNet=('dnet', 'median'), maxNet=('dnet', 'max'), ikiYarı=('ikiYarı', 'sum')).round(0).to_string())
c_ = ['v', 'ADX', 'stop', 'yön', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']
print('NET EN İYİ 8:'); print(R.sort_values('dnet', ascending=False)[c_].head(8).round(1).to_string(index=False))
print('MaxDD BAZI AŞMAYAN, İKİ YARI ARTI, NET EN İYİ 8:'); print(R[(R.MaxDD <= B['MaxDD'] + 0.01) & R.ikiYarı].sort_values('dnet', ascending=False)[c_].head(8).round(1).to_string(index=False))
