# CYBORG | 1000PEPEUSDT.P 5M SWING

BINANCE:1000PEPEUSDT.P, 5 dk, iki yön, sabit 500$ birim (initial_capital=500, komisyon %0.04, slippage 1 tick, process_orders_on_close).
Canlı aday: **V11.7** (TV 2026-11-21: 1751 işlem, net 9766.63$ = %1953, PF 1.69, MaxDD 306/318).

## Piyasa verisi (repo `data/`)

| Dosya | İçerik | Aralık |
|---|---|---|
| `data/1000PEPEUSDT_1m_FULL_20230505_THROUGH_20261119.csv.gz` | PEPE 1m OHLCV (UTC) | 2023-05-05 → 2026-11-19 |
| `data/1000PEPEUSDT_OI_5m_UTC_20230505_THROUGH_20261119.csv.gz` | PEPE OI + top trader / taker oranları, Binance metrics 5m | 2023-05-05 → 2026-11-19 |
| `data/BTCUSDT_OI_5m_UTC_20200910_THROUGH_20261119.csv.gz` | BTC OI + oranlar, Binance metrics 5m | 2020-09-10 → 2026-11-19 |
| `data/BTCUSDT_PERP_1m_2019_20261115.csv.gz` | BTC 1m OHLCV (BTC 4H motoru) | 2019 → 2026-11-15 |
| `data/1000PEPEUSDT_funding_events_2023-05_to_2026-11-20.csv` | PEPE funding olayları | 2023-05 → 2026-11-20 |

```python
import pandas as pd
R = 'https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/data/'
px = pd.read_csv(R + '1000PEPEUSDT_1m_FULL_20230505_THROUGH_20261119.csv.gz')   # timestamp_utc, open_time_ms, open, high, low, close, volume, ...
oi = pd.read_csv(R + '1000PEPEUSDT_OI_5m_UTC_20230505_THROUGH_20261119.csv.gz')  # timestamp_utc, open_interest_qty, open_interest_usdt, ...
boi = pd.read_csv(R + 'BTCUSDT_OI_5m_UTC_20200910_THROUGH_20261119.csv.gz')
```
OI zaman damgası bar sonu etiketli (`end_labeled_unchanged_ASSUMED`); motorda bar açılışına ffill + 1 bar gecikme (Pine: `request.security(OI, "5", close)[1]`).

## Proje dosyaları

- `pine/` — tüm sürümler (V2 → V12). Soketlerde bot kodu `BOTKODU` ile gizlendi (repo herkese açık); canlıda kendi kodunu yaz.
  - `USER_V11_7_WT_SOCKETS.pine` — kullanıcının canlı soketli V11.7 dosyası (kod gizli)
  - `USER_V11_2_WT_SOCKETS_INSTANT_TRAIL_REV29.pine` — kullanıcının V11.2 + anlık trail görseli
- `tv_exports/` — TradingView Strategy Tester export'ları (`TV_{sürüm}_{tarih}_{hash}.xlsx`).
- `engine/` — araştırma motoru (numba, TV-sadık: dolum kapanış ± 1 tick, iki bacak komisyon, HTF lookahead_on+[1]). Ana dosyalar: `anti.py` (run_v14, deb), `btc2h.py` (RUN5/chainY), `grand*.py` (geniş arama → V11.5), `cbrk*.py` (zarar serisi kesici → V11.6), `joint*.py` (BTC+PEPE ortak OI → V11.7), `instant*.py` (WunderTrading anlık TP/SL 1m simülasyonu), `oiab.py` / `oiuse.py` (OI katkı denetimi, 30 dk MTM DD), `shsl*.py` (short SL testi), `rollf*/rolls*/rollb*/breaker/ptgt/wfloor/wgrid/vhf/oi30.py` (reddedilen filtre/kilit denemeleri).
- `engine_data/` — türetilmiş küçük girdiler: `pepe_{5,30,60,240}min.parquet`, `LIB.pkl` (özellik kitaplığı), `btcOI.pkl`, `btcF.pkl`, `flowF.pkl`, `BC.pkl`. Büyük önbellekler (PR.pkl 1.2 GB, fixRES, msEQ) yüklenmedi; motor yeniden üretir.
- `charts/` — eski sürüm equity/DD PNG'leri.

## Durum

| Sürüm | Değişiklik | TV işlem | TV net $ | TV PF | MaxDD (kapanış) |
|---|---|---|---|---|---|
| V2 (ADX-MOM) | ilk sürüm | 209 | 4833 | 2.05 | 380 |
| V4 COLLIDE | çarpışma motoru | 681 | 4731 | 1.65 | 315 |
| V6 / V6.1 | | 757 / 1074 | 6142 / 6157 | 1.81 / 1.74 | 315 / 307 |
| V6.3 BTC | BTC 4H motoru | 1023 | 7817 | 1.70 | 326 |
| V8 / V8.1 | | 1197 | 7994 / 7930 | 1.67 / 1.68 | 361 / 272 |
| V9 | | 1867 | 8114 | 1.52 | 354 |
| V10 / V10.1 | | 1949 / 1898 | 8249 / 8378 | 1.51 / 1.53 | 260 |
| V11 / V11.1 | | 2093 / 2122 | 8576 / 8706 | 1.52 / 1.52 | 258 / 308 |
| V12 | | 2255 | 8418 | 1.49 | 329 (iptal) |
| V11.2 | mikro/orta motor 2 mum teyit (flip-flop temizliği) | — | motor 8927 | motor 1.54 | |
| V11.3 / V11.4 | short PF kapısı / haftalık kâr kilidi | — | iptal | | |
| V11.5 | geniş arama: mMomLimit 4.0, MID 3 mum teyit, liqSL 8 | — | motor 9672 | motor 1.67 | |
| V11.6 / V11.6_WT | gölge mimarisi + zarar serisi kesici; WT soketleri | 1733 | 9582 | 1.68 | 306 / 318 çubuk içi |
| **V11.7** | V11.6 + BTC & PEPE ortak OI (tükeniş kilidi + takip motoru) | **1751** | **9767** | **1.69** | **306 / 318 çubuk içi** |

TV sayıları export'lardan; V11.2 / V11.5 sadece araştırma motoru. `TV_ADX-MOM_SWING_V6_..._471b8` ve `TV_SWING_V10_..._9ef18` kısa pencere export'ları (son 1–3 ay).

Reddedilenler (motorda bazı geçmedi): kayan PF/Calmar filtresi, short PF kapısı, MaxDD kesici, haftalık/aylık kâr kilidi, VHF, 30 dk / 1 sa OI filtresi, 2 sa BTC OI, Pine içi short SL (−%20 zarar hedefi).
WunderTrading anlık ayar önerisi: +%6.5 / %0.10 geri çekilme, SL kapalı (düzenlilik için; saf net için anlık kural yok).
