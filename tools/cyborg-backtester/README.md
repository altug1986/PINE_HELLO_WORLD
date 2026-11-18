# CYBORG Backtester

Run Pine Script v5/v6 strategies on local OHLCV and compare closed trades with a TradingView Strategy Tester export. Files stay local during the simulation.

## Kurulum

Python 3.11 veya üstü ve Node.js 20 veya üstü gerekir. Aşağıdaki komutları bu dizinde çalıştır.

```sh
npm ci
python3 -m pip install -r requirements.txt
python3 selftest.py
```

XLSX okumak için openpyxl gerekir. CSV ve ZIP CSV akışı Python standart kütüphanesini kullanır.

## Çalıştırma

`config.example.json` dosyasını kopyala. Pine ve veri yolları yapılandırma dosyasının bulunduğu dizine göre çözülür. Tam yolu da verebilirsin. Yeni Pine yazacaksan `strategy.example.pine` başlangıç örneğidir.

```sh
python3 backtest.py inspect-tv TradingView.xlsx --tv-timezone Europe/Istanbul
python3 backtest.py run config.json --out run-baz --tv TradingView.xlsx --tv-timezone Europe/Istanbul
python3 backtest.py compare run-baz/trades.csv --tv TradingView.xlsx --out tv-karsilastirma.json
```

- Ana `timeframe` ve veri `timeframe` alanında TradingView biçimini kullan. Örnekler `"1"`, `"6"` ve `"60"`. 24/7 UTC dakika ızgarası ve günlük mum desteklenir. UTC ızgarası piyasanın seansıyla örtüşmüyorsa bu sürümü kullanma.
- `start` ve `end` ilk ve son grafikteki **mum açılış zamanlarıdır** ve dahildir. Ana Pine bu geçmişin ilk mumundan itibaren yürür. TradingView Deep Backtest başlangıcını, warm-up geçmişini ve açılış zamanını aynı şekilde eşleştir.
- CSV için zaman, open, high, low, close başlıkları gerekir. Hacim sütunu yoksa veri denetiminde işaretlenir. Pine hacim veya hacim tabanlı bir gösterge kullanıyorsa eksik hacim çalıştırmayı durdurur. Başlıksız Binance kline sırası da okunur. Çok üyeli ZIP için `member` olarak tam iç CSV adını ver. OHLC XLSX için varsayılan sayfa aktif sayfadır. `member` ile seçebilirsin. Zaman damgası sütunu mum açılışını göstermeli. Epoch saniye, ms, µs, ns ve ISO tarihleri desteklenir. Ofsetsiz OHLC zamanı için `feeds[].timezone`, ofsetsiz export için `--tv-timezone` kullanılır.
- Sembol bilgilerini borsa/ürün değerleriyle doldur. Her dış seri ayrı bir `feeds` kaydı ister. Birden fazla aynı sembol kaynağı farklı zaman dilimleriyle eklenebilir. Tam zaman dilimi mevcutsa o veri kullanılır. Diğer durumda tam bölünebilen en yakın alt zaman diliminden, yalnız tamamlanmış UTC mumlar birleştirilir. Küçük zaman dilimi dosyası büyük zaman diliminden türetilmez.
- Önceki HTF geçmişi kaynakta varsa okunur. Büyük dosyalarda her kaynak için `history_start` vererek normalizasyona giren geçmişi sınırla. İndikatörün ihtiyaç duyduğu ısınma payını bırak. Veri boşluğu giderildiyse aralık boyunca kesintisiz girdiyi tercih et. `allow_gaps: true` bilinçli bir seçimdir ve boş mum üretmez.
- `inputs` Pine input adlarını/değerlerini ve `properties` TradingView Strategy Properties anahtarlarını/değerlerini alır. Örnek `"properties": {"commission_value": 0.04, "slippage": 1}`. Komisyon birimi yüzde tipi için kaynak kodda `strategy.commission.percent` olmalı. Bağlı stratejinin özelliklerini aynı exporta göre ayarla. Her varyantı yeni çıktı dizinine yaz.

## Sonuç

- `trades.csv` kapalı işlem bacaklarını ve dolumları verir. `equity.csv` bar bazında motor equity değerini verir. `report.json` etkili motor ayarlarını, veri/Pine özetini, metrikleri ve varsa eşleşme sonucunu kaydeder.
- İşlem karşılaştırması yön, giriş/çıkış zamanı, iki dolum fiyatı, miktar ve net PnL üzerinden yapılır. Saat damgası toleransı sıfırdır. Net PnL toleransı iki ondalık haneli export için 0,005001 para birimidir. Açık işlemler ve dolmamış emirler eşleşme kapsamı dışındadır.
- `matched_closed_trades` sadece karşılaştırılan kapalı işlemlerin eşleştiğini gösterir. `mismatch` karşılaştırmada fark bulduğunda döner ve rapor yazılır. `no_tradingview_reference` yerel araştırma çıktısıdır. Süreç kodları 0 çalıştı, 1 hata, 2 exportla kapalı işlem eşleşmesi sağlanamadı şeklindedir.
- Sharpe ve Sortino sabit sermayeye bölünen takvim günü kapalı PnL'si üzerinden 365 gün ile yıllıklaştırılır, risksiz oran sıfırdır. Calmar doğrusal yıllık getiri / 10 dk MaxDD'dir. CV günlük/haftalık PnL standart sapması / mutlak ortalamadır. P95/P98/P99 10 dk ve saatlik kapalı PnL eğrisinde ölçülen drawdown derinliklerinin yüzdelikleridir. Metrik tanımsızsa `null` döner.
- PineTS, TradingView ile aynı çalışma zamanı değildir. Seanslı piyasalar, iç içe veri istekleri, Pine import kütüphaneleri, Bar Magnifier ve `request.security`/`request.security_lower_tf` dışındaki Pine veri çağrıları bu sürümün kapsamı dışındadır. Emir grupları, kısmi çıkış, MTF davranışı ve işlem içi dolumlar stratejiye özel TradingView exportuyla denetlenmelidir. Eksik kaynak için seri tahmin edilmez.

[PineTS kaynak kodu ve lisansı](https://github.com/LuxAlgo/PineTS)
