# Video dersi: yatayı sıfırlayamazsın → farklı hızda düşük korelasyonlu alt motorlar, AYNI 500$ bölünür (birim büyümez)
exec(open('/home/claude/zecwork/hold1.py').read().split("rows = []")[0].replace("RGi = np.nan_to_num(RG).astype(np.int64); ", ""))
RGi = np.nan_to_num(RG).astype(np.int64)
B0 = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/BOB_1DK_2HAFTA.csv'); H0 = pd.read_csv('/home/claude/pine_hello_world/ZEC/research/HOLD_1DK.csv')
P = pd.concat([B0[(B0.IS > 0) & (B0.OOS > 0)].assign(mod=0, k4=3), H0[(H0.IS > 0) & (H0.OOS > 0)]])
P = P.assign(sk=P.net / P.DDp98 * P.R2).sort_values('sk', ascending=False)
P = P.groupby(['tH', 'ts', 'hz', 'kS', 'mod']).head(2).head(48)           # hız/tip çeşitliliği
E = []; meta = []
for _, r in P.iterrows():
    L, S = select(r.thr, int(r.hz)); aL = (L >= r.kL); aS = (S >= r.kS) & ~aL
    eq, T = z03.sim10(o, h, l, c, a, aL, aS, float(r.ts), int(r.tH), spk, 60, RGi, A4 * r.k4, int(r['mod']))
    E.append(eq[::30].astype(np.float32)); meta.append((int(r.thr), int(r.hz), int(r.kL), int(r.kS), int(r['mod']), int(r.k4), r.ts, int(r.tH), len(T)))
E = np.array(E); m30 = mid // 30; b30 = blk // 30
W30 = {k: (v[0] // 30, min(v[1] // 30, E.shape[1] - 1)) for k, v in WIN.items()}
def sc(x, a_=0, b_=None):
    x = x[a_:b_]; dd = np.maximum.accumulate(x) - x; r2 = np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2
    return (x[-1] - x[0]) / max(np.percentile(dd, 98), 1) * r2
C = np.corrcoef(np.diff(E[:, ::48], axis=1)); print('ort. günlük korelasyon havuz:', round(C[np.triu_indices(len(E), 1)].mean(), 2))
out = []
for K in (2, 3, 4, 5, 6):                       # açgözlü seçim YALNIZ ilk yarıda
    S_ = [int(np.argmax([sc(e, 0, m30) for e in E]))]
    while len(S_) < K:
        S_.append(max((j for j in range(len(E)) if j not in S_), key=lambda j: sc(E[S_ + [j]].mean(0), 0, m30)))
    x = E[S_].mean(0); dd = np.maximum.accumulate(x) - x; bp = np.diff(x[b30])
    r = dict(K=K, net=x[-1], IS=x[m30] - x[0], OOS=x[-1] - x[m30], MaxDD=dd.max(), DDp95=np.percentile(dd, 95), DDp98=np.percentile(dd, 98), DDp99=np.percentile(dd, 99),
             R2=np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2, blokPoz=(bp > 0).mean() * 100, blokMed=np.median(bp), n=sum(meta[j][-1] for j in S_) / K,
             kor=np.corrcoef(np.diff(E[S_][:, ::48], axis=1))[np.triu_indices(K, 1)].mean())
    for k_, (a_, b_) in W30.items(): r[k_] = x[b_] - x[a_]
    r['motorlar'] = ' | '.join(str(meta[j][:8]) for j in S_); out.append(r)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 200)
R = pd.DataFrame(out); print(R.drop(columns='motorlar').round(2).to_string(index=False)); print(R.motorlar.to_string())
j = int(np.argmax([sc(e, 0, m30) for e in E])); x = E[j]; dd = np.maximum.accumulate(x) - x
print('Tek en iyi (IS seçim):', meta[j], 'net', round(x[-1]), 'MaxDD', round(dd.max()), 'DDp98', round(np.percentile(dd, 98)), {k_: round(x[b_] - x[a_]) for k_, (a_, b_) in W30.items()})
np.save('/home/claude/zecwork/ens_E.npy', E); pd.DataFrame(meta).to_csv('/home/claude/zecwork/ens_meta.csv', index=False)
R.to_csv('/home/claude/pine_hello_world/ZEC/research/ENS_1DK.csv', index=False)
