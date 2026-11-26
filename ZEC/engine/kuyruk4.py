# YÖN BAZLI KESİCİ: aynı yönde art arda N zarar (& ≥ X $) → SADECE o yön kapanır (diğer yön oynar); o yönün gölge işlemi kazanınca / T saat sonra açılır.
exec(open('/home/claude/zecwork/kuyruk2.py').read().split("base = np.ones(")[0])
def runD(N, X, md, P):
    K = len(EI); sel = np.ones(K, np.bool_); blk = {1: -1, -1: -1}; sN = {1: 0, -1: 0}; sL = {1: 0.0, -1: 0.0}; nc = 0
    for k in range(K):
        d = int(DR[k])
        if blk[d] >= 0 and ((md == 'T' and EI[k] - blk[d] >= P) or EI[k] - blk[d] >= 7 * 1440): blk[d] = -1
        sel[k] = blk[d] < 0; pn = PN[k]
        if sel[k]:
            if pn < 0: sN[d] += 1; sL[d] += pn
            else: sN[d] = 0; sL[d] = 0.0
            if sN[d] >= N and sL[d] <= -X: blk[d] = XI[k]; sN[d] = 0; sL[d] = 0.0; nc += 1
        elif md == 'W1' and pn > 0: blk[d] = -1
    return sel, nc
mB, wB, eB = ev(np.ones(len(EI), np.bool_)); out = []
for N, X, (md, P) in itertools.product((2, 3, 4), (0.0, 25.0, 50.0, 75.0), (('W1', 0), ('T', 360), ('T', 720), ('T', 1440))):
    sel, nc = runD(N, X, md, P); m, w, _ = ev(sel)
    out.append((N, X, md + (str(P) if md == 'T' else ''), m['net'] - mB['net'], m['IS'] - mB['IS'], m['OOS'] - mB['OOS'], m['MaxDD'], m['DD99'], m['Calmar'], w, nc, int((~sel).sum()), PN[~sel].sum()))
d = pd.DataFrame(out, columns=['N', 'X', 'dönüş', 'dNet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Calmar', 'enKötüSeri', 'kesme', 'atlanan', 'atlananNet']); d.to_csv('kuyruk4.csv', index=False)
g = (d.dIS > 0) & (d.dOOS > 0) & (d.MaxDD <= mB['MaxDD']); print('varyant', len(d), '| pozitif', int((d.dNet > 0).sum()), '| iki yarı + & MaxDD ≤ baz', int(g.sum()), '| kapı', int((g & (d.dNet >= 0.06 * mB['net'])).sum()))
pd.set_option('display.width', 250); print(d.sort_values('dNet', ascending=False).head(8).round(1).to_string(index=False))
L = PN; nx = np.r_[PN[1:], np.nan]; print('baz: zarar sonrası işlem ort. $', round(float(np.nanmean(nx[L < 0])), 2), '| kazanç sonrası', round(float(np.nanmean(nx[L > 0])), 2))
for N in (2, 3, 4):
    r = [PN[k] for k in range(N, len(PN)) if all(PN[k - j] < 0 for j in range(1, N + 1))]; print(f'{N} zarar sonrası sonraki işlem: n {len(r)} ort {np.mean(r):+.2f} kazanma %{np.mean(np.array(r) > 0) * 100:.0f}')
