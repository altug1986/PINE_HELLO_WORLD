# CYBORG | ZEC 2M — sistem organizasyonu

`BINANCE:ZECUSDT.P` · 2 dk · iki yön · sabit 500 $ birim (initial_capital 500, strategy.cash 500, komisyon %0,04, kayma 1 tick, process_orders_on_close, pyramiding 0, margin 0).
Durum: **V0.1 Pine hazır, TV export bekliyor** (2026-11-24). Sayılar araştırma motoru (`engine/z01.py`), TV'de doğrulanmadı.

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
| `ZECUSDT_1m_FLOW_UTC_{yıl}.csv.gz` ×6 | 1 dk taker alış (ZEC/USDT), işlem sayısı, quote hacim — 1 dk OHLC ile aynı `time` | 2021-01-01 → 2026-11-23 | eksik dosya 0/93 |
| `ZECUSDT_INDEX_PRICE_1m_UTC_…` | Binance index fiyatı 1 dk kapanış → basis = vadeli / index − 1 | aynı | basis P1/P50/P99 −28,7 / −3,9 / +28,4 bps |
| `ZECUSDT_PREMIUM_INDEX_1m_UTC_…` | premium index 1 dk kapanış (funding'in hammaddesi) | aynı | basis ile kor. 0,85 |
| `ZECUSDT_BOOKDEPTH_1m_UTC_{yıl}.csv.gz` ×4 | emir defteri: fiyatın ±%1–5 (2026-01-15'ten ±%0,2) içinde bekleyen alış/satış toplamı USDT, dakikanın son anlık görüntüsü (`snap_s`) | 2023-01-01 → 2026-11-23 | 1.420 gün, hatalı 0 |
| `ZECUSDT_WHALE_PRINTS_1m_UTC_{yıl}.csv.gz` ×7 | balina printleri: taker emri ≥10k/50k/200k/1M USDT adet + tutar, yön, en büyük emir (seyrek: yalnız print olan dakikalar) | 2020-02 → 2026-11-23 | 104 dosya, hatalı 0, 9,45 M emir |
| `candidates_1h/ZECUSDT_1h.csv.gz` | coin seçim aşamasının 1 s verisi | — | — |

Güncelleme: `.github/workflows/zec_1m.yml`, `zec_oi.yml`, `zec_micro.yml` (workflow_dispatch). Edge araştırma kütüğü: `research/EDGE_LOG.md` (skill: `cyborg-micro-edge`). Yükleyici: `ZEC/engine/zec_data.py` → `load(tf_min=2, lag=0)`.

### 2.1 MASTER 1m — tam senkron tek tablo
`data/ZECUSDT_MASTER_1m_SYNC_{2021…2026}.parquet` (56 kolon, 3.093.120 satır; üretici `engine/build_master.py`, denetim `data_quality/MASTER_SENKRON_DENETIM.json`). Okuma: `zec_data.load(tf_min=2, lag=0)`.
Satır = 1 dk bar, `time` = açılış (UTC ms); **satırdaki her değer bar kapanışında (T = time + 60 s) bilinen bilgidir.**

| Kolon grubu | Kaynak | Bilindiği an kuralı | Kapsam |
|---|---|---|---|
| `open…volume`, `quote_volume`, `trades`, `taker_buy_*`, `taker_delta_base` | 1m kline | aynı bar | 2021-01 → 2026-11-23 |
| `index_close`, `basis_bps`, `premium_bps` | index / premium 1m | aynı bar | aynı |
| `funding_pct` | funding olayları | son olay ≤ T (9 s tolerans) | → 2026-11-01 |
| `oi_zec`, `oi_usdt`, `tt_acc_ls`, `tt_pos_ls`, `glb_ls`, `taker_ls_5m`, `btc_oi`, `xmr_oi` | metrics 5m | damga ≤ T − 5 dk (`metrics_age_s` = T − damga, P50 420 s / P99 540 s) | ZEC/XMR 2021-12 →, BTC OI → 2026-11-19 |
| `book_m5…m1`, `book_m02`, `book_p02`, `book_p1…p5` | bookDepth | son anlık görüntü < T (`book_age_s`; %99,5'i aynı bar içinde) | 2023-01 →; ±0,2 2026-01-15 → |
| `wh_{buy,sell}_{max,n10k…q1000k}` | aggTrades (aynı ms + yön = bir taker emri) | aynı bar; print yoksa 0 | 2021-01 → |
| `btc_close`, `xmr_close` | BTC/XMR 1m | aynı bar | 2021-01 → 2026-11-23 |

TF'ye toplama: OHLC standart, akış ve balina adet/tutar **toplam**, `wh_*_max` **maks**, durum serileri **son değer**. `lag=k` metrics serilerini k × 5 dk ek geciktirir.

Senkron kanıtı (denetim dosyası): taker alış > hacim 0 dakika · balina alışı taker alış tutarını aşan dakika %0,19 · balina alışı ↔ taker alış korelasyonu gecikme 0'da 0,984 (±1: 0,55 / 0,54) · ZEC ↔ BTC ve ZEC ↔ XMR 1 dk getiri korelasyonu yalnız gecikme 0'da tepe (0,53 / 0,39; komşular < 0,04) · ΔOI son etiket testi (aşağıda).

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
| V0.1 | motor | 1.367 (%273) | 1.037 | 1,30 | 291 (P99 236) | 0,011 | SPIKE FADE çift yön; long +1.549 / short −182; IS 915 / OOS 452; `pine/ZEC_2M_V0.1.pine` |
