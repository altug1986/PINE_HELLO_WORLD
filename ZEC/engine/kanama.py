# KANAYAN BÖLGELER: V3.0'ın düşüş (kanama) dönemlerini çıkar → fiyat hareketi yönü (salınım yapısı) + piyasa yapıcının fiyatı süpürülmemiş likidite öbeğine (eşit tepe/dip kümesi) sürmesi
# kuralları SADECE kanama dönemlerinde (ilk yarı) seçilir → sonra ikinci yarı kanamaları ve GENEL sonuç.
exec(open('/home/claude/zecwork/adxfilt.py').read().split("dv = TR[:, 2].astype(int); rows = []")[0])
@njit(cache=False)
def rec30(o, h, l, c, tick, vpos, vxp, vnew, evt, H, stp):
    """V3.0 (Pine sırası) gerçek pozisyon yolu: rp bar sonu pozisyon, rx bar içi çıkış fiyatı, rn yeni giriş."""
    n = len(c); gb = 0; real = 0; ep = 0.0; sa = False; sb = 0; ss = 0.0; blk = 0; rp = np.zeros(n, np.int8); rx = np.zeros(n); rn = np.zeros(n, np.bool_)
    for i in range(1, n):
        if gb != 0 and gb == vpos[i - 1] and vxp[i] > 0: gb = 0
        if real != 0:
            if sa and real == 1 and l[i] <= ss: rx[i] = min(o[i], ss) - tick; real = 0; sa = False; blk = sb + H
            elif (not sa) and real == vpos[i - 1] and vxp[i] > 0: rx[i] = vxp[i]; real = 0
        v = vpos[i]
        if vnew[i]: gb = v
        elif gb != v: gb = 0
        if sa and i - sb >= H: sa = False
        if (not sa) and evt[i] and i >= blk and gb == 0 and real == 0: sa = True; sb = i
        tg = 1 if sa else gb
        if tg != real:
            real = tg
            if tg != 0: rn[i] = True; ep = c[i] + tick * tg
            if sa and tg == 1: ss = ep * (1.0 - stp)
        rp[i] = real
    return rp, rx, rn
rp, rx, rn = rec30(o, h, l, c, TICK, vpos, vxp, vnew, EVT, 1440, 0.08)
@njit(cache=False)
def liq(h, l, c, Lb, Rb, w):
    """Süpürülmemiş salınım tepe/dipleri (pivot, Rb bar sonra teyit), en yakın üst/alt seviye, %w bant içindeki küme sayısı, salınım yapısı yönü (YT+YD=+1, AT+AD=−1)."""
    n = len(c); W = Lb + Rb + 1; HA = np.zeros(400); LA = np.zeros(400); nh = 0; nl = 0
    nH = np.full(n, np.nan); nL = np.full(n, np.nan); kH = np.zeros(n, np.int16); kL = np.zeros(n, np.int16); pa = np.zeros(n, np.int8)
    ph0 = np.nan; ph1 = np.nan; pl0 = np.nan; pl1 = np.nan
    for i in range(W, n):
        j = 0
        for t in range(nh):
            if HA[t] > h[i]: HA[j] = HA[t]; j += 1
        nh = j; j = 0
        for t in range(nl):
            if LA[t] < l[i]: LA[j] = LA[t]; j += 1
        nl = j; p = i - Rb; mx = h[p]; mn = l[p]; isH = True; isL = True
        for t in range(i - W + 1, i + 1):
            if h[t] > mx: isH = False
            if l[t] < mn: isL = False
        if isH and (p == i - W + 1 or h[p - 1] < mx):
            if nh == 400:
                for t in range(399): HA[t] = HA[t + 1]
                nh = 399
            HA[nh] = mx; nh += 1; ph0 = ph1; ph1 = mx
        if isL and (p == i - W + 1 or l[p - 1] > mn):
            if nl == 400:
                for t in range(399): LA[t] = LA[t + 1]
                nl = 399
            LA[nl] = mn; nl += 1; pl0 = pl1; pl1 = mn
        if nh > 0:
            m = 1e18
            for t in range(nh): m = min(m, HA[t])
            nH[i] = m; k = 0
            for t in range(nh):
                if HA[t] <= m * (1 + w): k += 1
            kH[i] = k
        if nl > 0:
            m = -1e18
            for t in range(nl): m = max(m, LA[t])
            nL[i] = m; k = 0
            for t in range(nl):
                if LA[t] >= m * (1 - w): k += 1
            kL[i] = k
        if ph1 > ph0 and pl1 > pl0: pa[i] = 1
        elif ph1 < ph0 and pl1 < pl0: pa[i] = -1
    return nH, nL, kH, kL, pa
@njit(cache=False)
def ovPA(o, h, l, c, tick, rp, rx, rn, vS, vL, cS, cL, nH, nL):
    """V3.0 yolu üstüne: vS/vL yeni short/long girişini engelle; cS/cL eldeki short/long'u kes, kesilen seviyeyi fiyat süpürünce (ve V3.0 aynı yönde ise) yeniden katıl."""
    n = len(c); real = 0; ep = 0.0; q = 0.0; rz = 0.0; eq = np.zeros(n); lv = 0.0; cut = 0; nc = 0
    for i in range(1, n):
        if real != 0 and real == rp[i - 1] and rx[i] > 0: rz += q * (rx[i] - ep) * real - 0.0004 * q * (ep + rx[i]); real = 0
        v = rp[i]; tg = real
        if rn[i]:
            cut = 0; tg = 0 if ((v < 0 and vS[i]) or (v > 0 and vL[i])) else v
        elif v == 0: tg = 0; cut = 0
        elif cut != 0:
            if (cut < 0 and h[i] >= lv) or (cut > 0 and l[i] <= lv): cut = 0; tg = v
            else: tg = 0
        elif real != v: tg = real if real == 0 else 0
        if real != 0 and real == v and cut == 0 and ((real < 0 and cS[i]) or (real > 0 and cL[i])):
            tg = 0; cut = real; lv = nH[i] if real < 0 else nL[i]; nc += 1
        if tg != real:
            if real != 0: xp = c[i] - tick * real; rz += q * (xp - ep) * real - 0.0004 * q * (ep + xp); real = 0
            if tg != 0: real = tg; ep = c[i] + tick * real; q = 500.0 / c[i]
        eq[i] = rz + (q * (c[i] - ep) * real - 0.0004 * q * ep if real != 0 else 0.0)
    return eq, nc
F0 = np.zeros(n, np.bool_); e30, _ = ovPA(o, h, l, c, TICK, rp, rx, rn, F0, F0, F0, F0, np.full(n, np.nan), np.full(n, np.nan)); print(f'V3.0 yol kontrol {e30[-1]:.1f} (9.891,6)')
# KANAMA dönemleri: 30 dk MTM zirve → dip düşüşü ≥ 50 $ (zigzag)
x = e30[::30]; pk = 0; tr_ = 0; EP = []; i0 = 0
for t in range(1, len(x)):
    if x[t] > x[pk]:
        if x[pk] - x[tr_] >= 50 and tr_ > pk: EP.append((pk * 30, tr_ * 30))
        pk = t; tr_ = t
    elif x[t] < x[tr_]: tr_ = t
if x[pk] - x[tr_] >= 50 and tr_ > pk: EP.append((pk * 30, tr_ * 30))
EP = np.array(EP); dep = np.array([e30[a] - e30[b] for a, b in EP]); isE = EP[:, 0] < mid
print(f'kanama dönemi {len(EP)} (ilk yarı {isE.sum()}, ikinci {(~isE).sum()}) · toplam kanama {dep.sum():.0f} $ (ilk {dep[isE].sum():.0f} / ikinci {dep[~isE].sum():.0f}) · ort süre {np.mean(EP[:, 1] - EP[:, 0]) / 1440:.1f} g')
lp = []
for a, b in EP:
    k = (np.arange(a, b + 1)); pos = rp[a:b + 1].astype(float); r = np.diff(np.log(c[a:b + 2])); lp.append((np.sum(pos * r[:len(pos)] < 0) / max(np.sum(pos != 0), 1), np.sign(np.log(c[b] / c[a])), np.mean(pos)))
lp = np.array(lp); print(f'kanamada fiyat yönü yukarı %{(lp[:, 1] > 0).mean() * 100:.0f} · ort net pozisyon {lp[:, 2].mean():+.2f} (− = short ağırlıklı) · pozisyonun fiyata ters olduğu dakika %{lp[:, 0].mean() * 100:.0f}')
def bleed(e): return np.array([(e[b] - e[a]) - (e30[b] - e30[a]) for a, b in EP])
rows = []; T0b = time.time()
for Lb in (30, 60, 120):
    nH, nL, kH, kL, pa = liq(h, l, c, Lb, Lb, 0.01); dH = (nH / c - 1) * 100; dL = (1 - nL / c) * 100
    for X, kmin, usePA in itertools.product((0.5, 1.0, 2.0, 3.0), (1, 2), (True, False)):
        nearH = np.nan_to_num((dH <= X) & (kH >= kmin), nan=False); nearL = np.nan_to_num((dL <= X) & (kL >= kmin), nan=False)
        sS = nearH & ((pa > 0) if usePA else True); sL = nearL & ((pa < 0) if usePA else True)
        for kural, (vS, vL, cS, cL) in (('veto', (sS, sL, F0, F0)), ('kes+süpürünce dön', (F0, F0, sS, sL)), ('veto+kes', (sS, sL, sS, sL))):
            e, nc = ovPA(o, h, l, c, TICK, rp, rx, rn, vS, vL, cS, cL, nH, nL); bl = bleed(e)
            r = ev(e, nc, 0, 0, pivot=Lb, mesafe=X, küme=kmin, PAşart=usePA, kural=kural); r['kanamaİS'] = bl[isE].sum(); r['kanamaOOS'] = bl[~isE].sum(); r['iyileşenEpİS'] = (bl[isE] > 1).mean() * 100; rows.append(r)
R = pd.DataFrame(rows); B = ev(e30, 0, 0, 0)
for k in ('net', 'IS', 'OOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar'): R['d' + k] = R[k] - B[k]
R.to_pickle('/home/claude/zecwork/kanama.pkl'); pd.set_option('display.width', 300); print(f'{len(R)} varyant {time.time() - T0b:.0f} sn')
c_ = ['pivot', 'mesafe', 'küme', 'PAşart', 'kural', 'işlem', 'kanamaİS', 'kanamaOOS', 'iyileşenEpİS', 'net', 'dnet', 'dIS', 'dOOS', 'MaxDD', 'DD99', 'Ulcer', 'Calmar']
print('1) SADECE İLK YARI KANAMALARINA GÖRE SEÇİLEN EN İYİ 10 → ikinci yarı kanaması ve GENEL:'); print(R.sort_values('kanamaİS', ascending=False)[c_].head(10).round(1).to_string(index=False))
print(R.groupby('kural').agg(medKanİS=('kanamaİS', 'median'), medKanOOS=('kanamaOOS', 'median'), medNet=('dnet', 'median'), maxNet=('dnet', 'max')).round(0).to_string())
