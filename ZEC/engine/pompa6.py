# AKILLI POMPA STOPU: sadece derin düşüş sonrası sıçramada (fiyat N dk zirvesinin ≥ X% altında) stop mesafesi en az %F
exec(open('/home/claude/zecwork/pompa5.py').read().split("R = {}")[0].replace("(flP if {v} == 5 else 0.0) * ep", "(flA[i] if {v} == 5 else 0.0) * ep").replace("byp, capL, capS, flP, oppN", "byp, capL, capS, flA, oppN").replace("def chainZ(flP):", "def chainZ(flP):\n    flP = np.asarray(flP, np.float64)"))
import time; t0 = time.time(); cs = pd.Series(cc)
for N in (1440, 2880):
    dd = (cs / cs.rolling(N, min_periods=1).max() - 1).values
    for X in (0.08, 0.10, 0.12, 0.15, 0.20):
        for F in (0.05, 0.06):
            e, pm = chainZ(np.where(dd <= -X, F, 0.0)); m = met(e, '')
            print(f'{N // 60} s zirveden ≥%{X * 100:.0f} düşük → taban %{F * 100:.0f}: Δnet {m["net"] - 11023.4:+6.1f} yarılar {m["IS"] - 5577.7:+6.1f}/{m["OOS"] - 5445.7:+6.1f} MaxDD {m["MaxDD"]:5.1f} DD99 {m["DD99"]:5.1f} | POMPA {len(pm)} net {pm[:, 5].sum():+6.1f} | {time.time() - t0:.0f}s', flush=True)
