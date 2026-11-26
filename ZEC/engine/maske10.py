exec(open('/home/claude/zecwork/maske9.py').read().split("\nS3, A3 = mk('3h')\n")[0])
import time
mn = ixh.values.astype('datetime64[m]').astype(np.int64); day = mn // 1440; mod = mn % 1440
def mkm(m):   # TV: gün içi bölünmeyen dilimler her UTC gününde 00:00'dan başlar, günün son barı kısa
    b = day * 1440 + (mod // m) * m; g = pd.DataFrame({'b': b, 'o': o_, 'h': h_, 'l': l_, 'c': cc}).groupby('b', sort=True).agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    ct = np.minimum(g.index.values + m, (g.index.values // 1440 + 1) * 1440); D = g.reset_index(drop=True)
    st = np.sign(D.c.ewm(span=10, adjust=False).mean() - D.c.ewm(span=30, adjust=False).mean()).values; ax = adx(D)
    j = np.searchsorted(ct, mn, side='right') - 1; ok = j >= 0; j = np.maximum(j, 0)
    return np.where(ok, st[j], 0.0), np.where(ok, ax[j], 0.0)
bn = B['net']; out = []; t0 = time.time()
for m in range(10, 730, 10):
    S, A = mkm(m)
    for z in (30.0, 35.0, 40.0):
        e, nE, _ = comb3(o_, h_, l_, cc, 0.01, rp, rx, S, A, z, 0.05); r = met(e, '')
        out.append((m, z, r['net'] - bn, r['IS'] - B['IS'], r['OOS'] - B['OOS'], r['MaxDD'], r['DD99'], r['Calmar'], nE))
    print(m, f'{time.time() - t0:.0f}s', [round(x[2]) for x in out[-3:]], flush=True)
d = pd.DataFrame(out, columns=['dk', 'adx', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'giris']); d.to_csv('maske10.csv', index=False)
