# Piyasa korku verisi

| Dosya | İçerik |
|---|---|
| `BTC_KORKU_ENDEKSI_1S_BINANCE.csv.gz` | Saatlik vekil korku endeksi (0 = aşırı korku, 100 = aşırı açgözlülük), 2020-12-10 → 2026-11-15 21:00 UTC. Yalnız Binance BTCUSDT.P verisi: 1 dk fiyat + 5 dk metrics (taker oranı, OI). Satır zamanı = saatin kapanışı (UTC); değer o saatin sonunda bilinir, bir sonraki saatte kullanılır. |
| `fear.py` | Üretici. Bileşenler, her biri geçmiş 90 günlük yüzdelik sıra (geleceğe bakmaz): 24 s oynaklık %25 (ters), fiyat / 30 g ort. %25, 24 s getiri %15, 90 g zirveden düşüş %20, 24 s taker alış/satış oranı %15. Metrics 1 bar gecikmeli. |
| `FNG_ALTERNATIVE_ME_GUNLUK_LE20.csv` | alternative.me günlük Fear & Greed, yalnız ≤20 olan günler, 2021-12 → 2026-11-22. Sunucudan indirilemedi; web sayfa okuyucusuyla satır satır çıkarıldı, programla doğrulanmadı. |

alternative.me, CoinMarketCap ve CoinGecko endeksleri yalnız günlük yayımlanır; saatlik geçmişleri yok. Vekilin alternative.me ile uyumu (günlük ortalama): FGI<15 günlerinde medyan 28, diğer günlerde 52, AUC 0,805.

Test (2026-11-23, araştırma motoru): vekil <10/15/20/25 iken XMR long girişlerini engellemek her eşikte kaybettirdi (engellenen işlemler toplam +347 … +1.246). O saatlerde portföyü kapatmak neti %5–21 düşürdü; MaxDD/Calmar komşu eşiklerde tutarsız. Karar: kullanılmadı.
