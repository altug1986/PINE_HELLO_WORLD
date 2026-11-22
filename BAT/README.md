# CYBORG BAT 5M KONSORSİYUM — veri ve çalışma arşivi

Binance USDT-M `BATUSDT.P`, 5 dk, sabit 500 $ birim (initial_capital 500, komisyon %0.04, kayma 1 tick).
Güncelleme: 2026-11-22.

## Piyasa verisi (`data/`)

| Dosya | İçerik | Aralık |
|---|---|---|
| `data/BATUSDT_1m_FULL_THROUGH_20261120_part1of2.csv.gz` | BAT 1 dk OHLCV, Binance kline sütunlarının hepsi | 2020-02-13 → 2023-06-30 |
| `data/BATUSDT_1m_FULL_THROUGH_20261120_part2of2.csv.gz` | aynı | 2023-07-01 → 2026-11-20 |
| `data/BATUSDT_OI_5m_UTC_THROUGH_20261120.csv.gz` | BAT açık pozisyon (OI), 5 dk, `durum` sütunuyla | 2021-12-01 → 2026-11-20 |
| `data/BATUSDT_OI_10m_UTC_THROUGH_20261120.csv.gz` | OI, 10 dk anlık gözlem | aynı |
| `data/BATUSDT_METRICS_5m_TOPTRADER_OI_20211201_20261120.csv.gz` | Binance günlük metrics zip'lerinin (1816 gün) birleşimi: OI, top trader L/S hesap (`count_toptrader_long_short_ratio`) ve pozisyon (`sum_toptrader_long_short_ratio`), genel L/S, taker al/sat | 2021-12-01 → 2026-11-20 (2022'nin çoğu boş) |
| `data/BTCUSDT_PERP_1m_2019_20261115.csv.gz` | BTC 1 dk (motorda kullanılan dosyanın aynısı, zaten depodaydı) | 2019-09-08 → 2026-11-15 |

1 dk dosyası 100 MB sınırı yüzünden iki parça; okurken birleştir:

```python
import pandas as pd
u = 'https://raw.githubusercontent.com/altug1986/PINE_HELLO_WORLD/main/data/'
bat = pd.concat([pd.read_csv(u + f'BATUSDT_1m_FULL_THROUGH_20261120_part{k}of2.csv.gz') for k in (1, 2)], ignore_index=True)
```

Zaman: `open_time_ms` / `create_time` UTC. Veri kalite raporları: `BAT/data_quality/`.

## Proje dosyaları (`BAT/`)

- `pine/` — tüm Pine sürümleri (SWING V1.0 → KONSORSİYUM V6.6, Astre V6.2, balina veri testi).
- `tv_exports/` — TradingView Strategy Tester export'ları (V5.4, V5.5, V5.6B, V5.9, Astre V6.2, V6.3 ×2) ve PEPE V11.7 karşılaştırma export'ları.
- `engine/` — offline TV-sadık numba motoru ve araştırma betikleri (`v55.py` katman, `eng.py`, `mtm30.py` 30 dk MTM metrikleri, `teco.py`, `astre_test.py`, `tvcmp63.py` TV eşleştirme, `stlib.py` Supertrend, `whale_code.json` gömülü balina kodu, `build_top.py` metrics birleştirici). Ara önbellekler (`*.pkl`, `*.npz`, ~1.3 GB) yüklenmedi; betikler veriden yeniden üretir.
- `data_quality/` — 1 dk eksik aralık / hatalı satır / indirilemeyen gün listeleri ve OI kalite raporu.

## Durum (2026-11-22, öğle)

| Sürüm | Kaynak | Net $ | İşlem | Not |
|---|---|---|---|---|
| V6.3 | TV | 8752 | 3018 | PF 1.415 |
| V6.7 TREND (Astre) | TV | 9326 | 3164 | PF 1.411, MaxDD 289/300 |
| V7.0 | TV | 9967 | 2968 | PF 1.478, MaxDD 262/275 |
| **V7.1** | **TV** | **10198** | 3162 | **PF 1.456, MaxDD 255/275, Calmar 30 dk 5.70, Sortino 3.56 — seçilen** |

Ayrıntılar, sorunlar ve dersler: `BAT/KAIZEN_2026-11-22.md`. Zaman dilimi parametreli, TV ile doğrulanmış motor: `BAT/engine/p1m.py` (yan motor için `run(D, ovr=...)`, ek kapı `xGL/xGS`, ek oy `xP/xw`); V6.7 trend katmanı portu `BAT/engine/rev25.py`.
