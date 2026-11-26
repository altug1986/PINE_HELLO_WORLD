exec(open('/home/claude/zecwork/maske10.py').read().split("bn = B['net']; out = []")[0])
bn = B['net']; rng = np.random.default_rng(11); NM = 150
for m, z in ((180, 40), (180, 38), (180, 37), (190, 38), (190, 37)):
    S, A = mkm(m); real = comb3(o_, h_, l_, cc, 0.01, rp, rx, S, A, float(z), 0.05)[0][-1] - bn
    cut = np.r_[0, np.flatnonzero(np.diff(S) != 0) + 1]; ln = np.diff(np.r_[cut, len(S)]); yon = []; zam = []
    for k in range(NM):
        Sm = np.repeat(rng.choice([-1.0, 1.0], len(cut)), ln) * np.abs(S); yon.append(comb3(o_, h_, l_, cc, 0.01, rp, rx, Sm, A, float(z), 0.05)[0][-1] - bn)
        sh = int(rng.integers(30 * 1440, len(S) - 30 * 1440)); zam.append(comb3(o_, h_, l_, cc, 0.01, rp, rx, np.roll(S, sh), np.roll(A, sh), float(z), 0.05)[0][-1] - bn)
    yon, zam = np.array(yon), np.array(zam)
    print(f'{m} dk ADX≥{z}: gerçek {real:+.0f} | yön maymunu medyan {np.median(yon):+.0f} P95 {np.percentile(yon, 95):+.0f} p={(yon >= real).mean():.3f} | zaman maymunu medyan {np.median(zam):+.0f} P95 {np.percentile(zam, 95):+.0f} p={(zam >= real).mean():.3f}', flush=True)
