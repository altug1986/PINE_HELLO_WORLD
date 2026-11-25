# Ek: baz FLAT iken MOMENTUM mikro (sıçrama yönünde gir); baz pozisyondayken mikro yok. Çıkış: momentum söndü (RSI10 geri), süre, stop, +%3 kâr.
src = open('/home/claude/zecwork/mikro.py').read(); src = src.split("if __name__ == '__main__':")[0]
src = src.replace("if tu[i] and fu[i]: md = -1\n                        elif td[i] and fd[i]: md = 1", "if tu[i] and fu[i]: md = 1\n                        elif td[i] and fd[i]: md = -1")
src = src.replace("ex = (R10[i] <= Y or i - mBar >= tb or c[i] >= mPx * 1.03) if mDir < 0 else (R10[i] >= 100.0 - Y or i - mBar >= tb or c[i] <= mPx * 0.97)",
                  "ex = (R10[i] <= 100.0 - Y or i - mBar >= tb or c[i] >= mPx * 1.03) if mDir > 0 else (R10[i] >= Y or i - mBar >= tb or c[i] <= mPx * 0.97)")
src = src.replace("if D > 0 and tu[i] and fu[i]: md = -1", "if False: md = -1").replace("elif D < 0 and td[i] and fd[i]: md = 1", "elif False: md = 1")
assert src.count("md = 1") >= 2; exec(src)
CB = np.array([(0, 0, 0, 0, 0, 0, 0)] + [(t, f, y, tb, s_, 1, 1) for t in range(len(TRG)) for f in range(len(FIL)) for y, tb, s_ in itertools.product(range(3), range(2), range(2))], np.int64)
R = grid(CB, o, h, l, c, posA, bXp, bXd, R10, TU, TD, FU, FD, YV, TBV, SV, TICK, mid, t23, t300, B6, b6, BM, bm)
df = pd.DataFrame(R, columns=['net', 'IS', 'OOS', 'yakın', 'son300', 'MaxDD', 'mddIS', 'mddOOS', 'Ulcer', 'blok+', 'blok', 'ay+', 'ay-', 'mikro', 'mikroNet', 'mikroPF', 'işlem'])
df['tetik'] = [TRG[k][0] for k in CB[:, 0]]; df['filtre'] = [FIL[k][0] for k in CB[:, 1]]; b = df.iloc[0]
for k in ('net', 'IS', 'OOS', 'MaxDD', 'Ulcer'): df['d' + k] = df[k] - b[k]
df.to_pickle('/home/claude/zecwork/mikro2.pkl'); v = df.iloc[1:]
print(f"baz {b.net:.1f} | FLAT MOMENTUM {len(v)} varyant | iki yarı artı %{((v.dIS>0)&(v.dOOS>0)).mean()*100:.1f} | medyan dNet {v.dnet.median():.0f} | max dNet {v.dnet.max():.0f} | mikro PF>1 %{(v.mikroPF>1).mean()*100:.1f} | medyan mikro adet {v.mikro.median():.0f} PF {v.mikroPF.median():.2f}")
print(v.sort_values('dnet', ascending=False)[['tetik', 'filtre', 'mikro', 'mikroPF', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'Ulcer']].head(5).round(2).to_string(index=False))
