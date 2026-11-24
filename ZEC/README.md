# CYBORG | ZEC 2M — sistem organizasyonu

`BINANCE:ZECUSDT.P` · 2 dk · iki yön · sabit 500 $ birim (initial_capital 500, strategy.cash 500, komisyon %0,04, kayma 1 tick, process_orders_on_close, pyramiding 0, margin 0).
Durum: **Faz 0 — kurulum** (2026-11-24). Henüz Pine / TV export yok; tüm sayılar araştırma motoru etiketli olacak.

## 1. Hedef

| | Tanım |
|---|---|
| Referans | LTC Trinity %40 + PEPE 5M %40 + BAT 5M %20 karışımının portföy getiri eğrisi (Altuğ'un "istediğim sonuç" örneği) |
| ZEC hedefi | Referansa göre **kâr ≥ 2×**, **risk ≥ 1,5× daha az**, karar metriği SUPER-CALMAR V4 (SC4) |
| Mutlak | Net **%3000** (500 $ → +15.000 $), overfitsiz |
| Kırmızı çizgi | MaxDD ve Calmar referanstan kötü olamaz; risk metrikleri 30 dk MTM (açık poz dahil) eğriden |
| Hedef tablo | Faz 0'da 3 TV export'tan sayısal tabloya çevrilir → `research/HEDEF_TABLO.md` |

Hedef tablo sütunları: net $ ve %, yıllık %, MaxDD, DD P95/P98/P99, Calmar, SC4, Sharpe, Sortino, Ulcer, σ10 dk / σ1 s, pozitif ay/hafta, en kötü hafta, işlem sayısı, 12 ay ileri MaxDD P95/P99.

## 2. Veri (hepsi `data/`, raw okunur)

| Dosya | İçerik | Aralık | Denetim |
|---|---|---|---|
| `ZECUSDT_1m_UTC_{yıl}.csv.gz` ×6 | 1 dk OHLCV (`time` = açılış, UTC ms) | 2021-01-01 → 2026-11-23 | 3.093.120 satır, tekrar 0, hatalı OHLC 0 |
| `ZECUSDT_METRICS_5m_UTC_20211201_THROUGH_20261123.csv.gz` | OI (ZEC/USDT), top trader hesap + pozisyon L/S, genel L/S, taker L/S, `flag_*` | 2021-12-01 → | 523.444 satır, 5 dk hizalı |
| `ZECUSDT_OI_5m_…` / `ZECUSDT_OI_10m_…` | OI tek başına, `durum` sütunlu | 2021-12-01 → | 198 / 98 BOŞ |
| `ZEC_TOP_TRADER_POSITIONS_5m_…` (+AUDIT, MISSING_DAYS) | top trader pozisyon L/S ham | 2021-12-01 → | 1 gün 404 (2023-12-13) |
| `ZECUSDT_PERP_TAKER_BUY_SELL_5m_…` | taker al/sat ZEC + USDT, delta, pay | 2021-01-01 → | hacim ve kapanış 1 dk ile %100 eşleşiyor |
| `ZECUSDT_FUNDING_EVENTS_20210101_20261123.csv` | funding olayları (8 s) | 2021-01-01 → **2026-10-31** | Kasım 2026 eksik |
| `candidates_1h/ZECUSDT_1h.csv.gz` | coin seçim aşamasının 1 s verisi | — | — |

Güncelleme: `.github/workflows/zec_1m.yml`, `zec_oi.yml` (workflow_dispatch). Yükleyici: `ZEC/engine/zec_data.py` → `load(tf_min=2, lag=0)`.

### Zaman kuralı (değişmez)
- **Metrics/OI damgası son etiketli:** ΔOI(t) en güçlü [t−5, t) hacmiyle eşleşiyor (Spearman 0,579; bir sonraki pencere 0,537; önceki 0,447). Bar kapanışı T'de yalnız **damga ≤ T − 5 dk** kullanılır (1 periyot güvenlik). Her OI kuralı 0/1/2 ek gecikmede ayrıca geçmeli.
- Taker 5 dk: pencere kapanışında bilinir (`available_from_utc`).
- Pine tarafı: `request.security("BINANCE:ZECUSDT.P_OI", "5", close, lookahead=barmerge.lookahead_on)[1]`; TV'deki OI geçmiş başlangıcı ilk export'ta "motorun kapattığı girişler hangi tarihten sonra kayboluyor" testiyle bulunur.

### Veri uyarıları
1. **İki boşluk:** 2022-02-26 → 03-01 (3 gün) ve 2022-04-01 → 04-03 (2 gün). Taker dosyasında da yok → Binance tarafında veri yok; doldurulmaz. Liste: `data_quality/1m_EKSIK_ARALIKLAR.csv`.
2. **Top trader 2022'de çoğunlukla boş** (çeyreklik kapsam %0–31); 2023Q1'den itibaren %100. Top trader özellikleri 2023+ ile sınırlı ya da kapalıyken baz aynı kalmalı. OI 2021-12'den tam.
3. **2021'de OI yok** (Ocak–Kasım) → OI'li kurallar o dönemde pasif; raporda ayrı gösterilir.
4. **Fiyat rejimi:** 2024 dibi 15,88 → 2026 tepesi 1.685 (~100×). Sadece % / ATR% ölçüler; sabit fiyat eşiği yasak. Kârın 2025–26'ya yığılma riski yüksek → zaman blok dağılımı ve TECO (mega kâr kümesi dışı) zorunlu.
5. 2021-05-19 13:31–13:37 −%19 / +%16 dakikalık fitiller gerçek (çöküş günü). 4.309 sıfır hacimli dakika.
6. Funding 2026-10-31'de bitiyor; yükleyici Kasım'ı boş bırakır. Funding kuralı denenmeden önce güncellenecek.

## 3. Mimari taslağı (Faz 2'de test edilir, kesin değil)
- **Çekirdek:** Trinity tipi çoklu motor oyu — A trend/momentum, B mean-reversion (z-skor), C BAT tipi ağırlıklı skor. Karşı oy yoksa tek motor yeter; gölge master histerezisi; motor başına ATR% çıkış.
- **Yan motorlar (öncelik zinciri, PEPE/BAT tipi, oy değil):** ZEC OI boşalması, likidasyon dönüşü, taker dengesizliği, funding ucu. Her biri tek tek eklenir, sıkı stop + teyit.
- **Rejim kapısı:** 15 dk / 1 s ADX + Supertrend, en az tutma süresi.
- **Long tarafı:** küçük XMR korelasyon filtresi (Altuğ'un isteği).
- **Taşınmayanlar:** saat/gün kuralları, PnL-bant governor, tarih/fiyat-seviyesi kapısı, `input.bool` aç/kapa.

## 4. Fazlar

| Faz | İş | Çıktı / kapı |
|---|---|---|
| 0 | Hedef tablo (3 referans export → 30 dk MTM karışım) | `research/HEDEF_TABLO.md` |
| 1 | Veri hattı: 2 dk bar, tick 0,01, özellik deposu (OI/taker/funding/XMR), zaman kuralı | `engine/zec_data.py` ✓ |
| 2 | Çekirdek A+B+C oy, filtresiz → hedef tabloyla kıyas | ilk baz (V0.1) |
| 3 | Yan motorlar tek tek | her biri bazdan ≥ %6 + risk kötüleşmez |
| 4 | XMR long filtresi | aynı kapı |
| 5 | Sağlamlık | aşağıdaki liste |
| 6 | Pine v6 tek dosya + Wunder soketleri → TV export → motor eşleşmesi | TV doğrulanmış sürüm, canlı aday |

Sağlamlık listesi: IS ilk yarı seçim → OOS ikinci yarı ve tersi · 6 aylık blokları tek tek dışarıda bırakma · komşu plato (±1 adım) · zaman dilimi kaydırma (1/2/3/4/5 dk, süre eşdeğeri) · OI gecikme 0/1/2 · MC bootstrap + perm · monkey (yön/zaman) · stres (%0,08 + 3 tick) · SC4 katman 2 (Robust P5, SCR) · 12 ay ileri MaxDD P90–P99.

## 5. Kabul kapısı (her revizyon)
Net ≥ baz × 1,06 **ve** MaxDD, Calmar, DD P95/P98/P99, SC4 kötüleşmez · σ10 dk / σ1 s düşük kalır · işlem başı anlık ters gidiş ~%15'e yaklaşmaz · bar içi hareketle ölçülür (derin ters gidip temiz kapanan işlem ödüllendirilmez) · tek değişiklik, baza karşı.

## 6. Klasörler
- `engine/` — yükleyici, motor, araştırma betikleri
- `pine/` — Pine sürümleri (`CYBORG | ZEC 2M … V0.x`)
- `tv_exports/` — Strategy Tester export'ları
- `research/` — hedef tablo, test raporları, grafikler
- `data_quality/` — `KALITE_RAPORU.json`, `1m_EKSIK_ARALIKLAR.csv`

## 7. Sürüm kütüğü

| Sürüm | Kaynak | Net $ | İşlem | PF | MaxDD (MTM) | SC4 | Not |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | Faz 0 |
