# CYBORG 4600 — nominal seçimi (2026-11-23)

Araştırma portföyü kararı. Canlı Wunder dağılımı değişikliği ayrı onay gerektirir. TV export işlem otoritesidir; aşağıdaki sayılar araştırma motoru (TV export işlemleri + Wunder Instant kuralları, 10 dk MTM, sabit nominal, funding hariç).

- Pencere: 2023-05-08 → 2026-11-14 (185.203 × 10 dk, 183 tam hafta). Path-Lock = V4.1 LIQ. DOGE short = V7 SERİ (7a533). BAT = V7.2 (v72).
- Son hali (09:02): Micro Ana 280, Micro Partial 150 → toplam **4.595** (hedef 4.600'ün 5 altı). Long ÷ Short (BTCDOM short sayılır): baz 1,10 → seçim 1,02. Nominaller: NOMINAL.tsv. Yüzdeler kendi nominaline göre.

| | Baz | Seçim |
|---|---|---|
| Net | 26.547 | 26.930 |
| MaxDD (% kendi nominali) | 3,47 | 3,31 |
| DD P95 / P99 | 1,29 / 1,94 | 1,25 / 1,90 |
| Calmar | 47,2 | 50,3 |
| Negatif hafta (183) | 11 | 7 |
| Son 90 / 300 gün net | 1.332 / 4.357 | 1.355 / 4.406 |

12 ay ileri MaxDD (% 4600, son 18 ay havuzu, haftalık blok MC 4000 tur): P50 2,62 · P90 3,45 · P95 3,76 · P99 4,61; kenar −%25 stres: P95 5,10 · P99 6,26 → teminat planlama tabanı ≈ %6,3.

Karşılaştırma adayları (aynı motor): Beton-500 (net 27.535, MaxDD 2,65, L/S 0,87), Beton-LS (net 26.772, MaxDD 2,69, 1 saat sapma en düşük, L/S 1,01). Kullanıcı L/S ≈ 1 ve son 90 gün kârını tercih etti.

İnteraktif regülatör (claude.ai artifact, sahibine özel): https://claude.ai/artifact/SQw6EPNX2XbCAHdxgKie9D — "Seçimim (23 Kas)" düğmesi.

## Canlı Wunder son hali (23 Kas 09:50) — tüm Instant ayarları dahil
Nominaller (Pine ×2): Path-Lock 300/95 · Micro 280/150 · MANU 226/226 · XMR Short 5dk 200 · 4dk 200 · 2dk 320 · DOGE Long 220 · DOGE 3dk 200 · BTCDOM 500 · LTC Trinity 370 · Polaris 210 · ETC 250 · ADA 200 · PEPE 250 · BAT 150 · DOGE V7 SERİ 250 = 4.597.
Değişen Wunder kuralları: DOGE Short SL10 + TP %2→%10 (eski %1,5→%40: −146 USDT); DOGE Long SL17 + TP %2→%10 + trailing 4%|0,5%; LTC Trinity SL10 + TP %4→%50 (modelde yoktu, sapmayı en çok düşüren). BAT SL15, XMR Short 4dk SL8 (c76c1 ile de), XMR Short 2dk SL8 hiç tetiklenmiyor (etki 0).

| Net | MaxDD | DD P95/P99 | Calmar | 10 dk sapma | 1 saat sapma | Neg. hafta | Son 90/300 |
|---|---|---|---|---|---|---|---|
| 26.350 | %3,21 | 1,24/1,85 | 50,7 | %0,0579 | %0,1457 | 7 | 1.337 / 4.391 |

12 ay ileri MaxDD (%): P50 2,58 · P90 3,39 · P95 3,71 · P99 4,49; avantaj aşınması −%10 (20.000 tur): P95 4,15 · P99 5,01 → teminat tabanı ≈ %5 (≈230 USDT); −%25 aşınmada P99 6,00 (≈276 USDT).

XMR Short 4dk 23 Kas 10:42 itibarıyla c76c1 "G5 Fiyat oku" (önceki 5e69d; aynı soket dd503…): kol 173,9 → 310,8 USDT (200 nominal); portföy net 26.213 → 26.350, Calmar 50,1 → 50,7.
