exec(open('/home/claude/zecwork/pompa8.py').read().split("out = []; t0 = time.time()")[0])
ok1 = np.ones(n, np.bool_); z0 = np.zeros(n, np.bool_); zf = np.zeros(n)
for eThr in (15.0, 20.0, 25.0):
    for dp in (0.025, 0.03, 0.035, 0.04, 0.05):
        rp_, rx_, Tb = chainW(eThr, dp); pm = Tb[(Tb[:, 7] == 5) & (Tb[:, 2] == 1)]
        m = met(combR(o_, h_, l_, cc, 0.01, rp_, rx_, S9, A9, 38.0, 0.05, z0, zf, zf, 720, 240, 0.0, 0, 0.05, ok1)[0], '')
        print(f'20 dk ADX≥{eThr:.0f} & −DI>+DI & girişten ≥%{dp * 100:.1f} aşağı → çık: Δnet {m["net"] - 11023.4:+6.1f} yarılar {m["IS"] - 5577.7:+6.1f}/{m["OOS"] - 5445.7:+6.1f} MaxDD {m["MaxDD"]:.1f} DD99 {m["DD99"]:.1f} | POMPA net {pm[:, 5].sum():+.1f} erken çıkış {(pm[:, 6] == 5).sum()}', flush=True)
