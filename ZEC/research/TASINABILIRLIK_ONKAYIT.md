# Taşınabilirlik (Edge Koruma Oranı) testi — ÖN KAYIT (sonuçlara bakmadan, 2026-11-25 06:15 TR)

Amaç: ZEC için kurulan sistemin ZEC karakterini mi ezberlediğini, yoksa piyasanın genel davranışını mı yakaladığını ölçmek. Monte Carlo istatistiksel dayanıklılığı, bu test piyasadan bağımsızlığı sınar.

## Donmuş olanlar (yeniden optimizasyon YASAK)
- Aday sistemler: ZEC-DENGELİ ve ZEC-NETMAKS (`ZEC/engine/mega5.py`), X1 referans.
- Bütün sayısal parametreler + ZEC ilk yarısında seçilmiş kural tablosu (bileşen × yön × rejim hücresi) aynen uygulanır. Diğer coinde kural seçimi yeniden YAPILMAZ.
- Maliyet: %0,04 × 2 bacak + 1 tick; sabit 500 $; 1 dk; açık poz dahil 30 dk MTM.

## Coin listesi (10, önceden sabit)
ETH, BTC, SOL, XMR, LTC, DOGE, ADA, ETC, BAT, PEPE (PEPE 2023-05'ten, kısa pencere olarak işaretli).
Bağımsızlık: test sırasında günlük getiri korelasyonundan etkin bağımsız coin sayısı hesaplanır; yüksek korelasyonlu kümeler (BTC/ETH/SOL, DOGE/PEPE, LTC/ETC/ADA) tek kanıt sayılır.

## Kalite skoru Q (0–10), bütün coinlerde aynı
Q = 10 × ortalama(8 alt skor, her biri 0–1'e kırpılır):
1. PF: (PF − 1) / 0,6
2. Sharpe (günlük MTM, yıllık): Sharpe / 2,5
3. Sortino: Sortino / 3,5
4. Calmar (yıllık net / MaxDD): Calmar / 4
5. UPI (yıllık getiri / Ulcer): UPI / 10
6. Haftalık CV: 1 − CV / 4
7. DD P99 (500 $'ın %'si): 1 − P99% / 80
8. İki yarı: ikisi pozitif 1 · biri 0,5 · hiçbiri 0

ERR = Q(yabancı) / Q(ZEC). Yorum: ≥%80 çok güçlü · %60–80 güçlü · %40–60 kısmi · %20–40 zayıf · <%20 ciddi genelleme sorunu.
Asgari araştırma eşikleri (her yabancı coin): PF ≥ 1,30 · Sharpe ≥ 1,5 · Calmar ≥ 2,0 · net > 0 · DD P95/P98/P99 raporlanır. Bunlar araştırma eşikleri, evrensel standart değil.
Hedef: ZEC Q ≈ 8 iken yabancı coinlerde Q ≈ 5 (ERR ≈ %60); 2/10 düzeyi kabul edilmez.

## Veri kısıtı (test öncesi bilinen)
Oy bileşenleri: OHLC, OI, top trader pozisyon, global L/S, basis, emir defteri, balina baskıları, BTC/XMR fiyatı.
Basis/emir defteri/balina şu an yalnız ZEC'te var; diğer coinlerde bu bileşenler 0 (oy vermez) kabul edilir ve sonuç "taşınabilir çekirdek" olarak etiketlenir. Tam test için 10 coinin metrics + bookDepth + aggTrades verisi Actions ile çekilecek.
