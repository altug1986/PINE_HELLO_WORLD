# CYBORG 4600 | OHLCV archive continuation — 2026-11-24

Source: Binance USDⓈ-M Futures /fapi/v1/klines, interval=1m. All timestamps UTC Unix milliseconds. Snapshot cutoff 2026-11-24 06:05 UTC (09:05 TRT); last closed candle open 06:04 UTC, close 06:04:59.999 UTC. No live/incomplete candle included.

Prior archive patch: commit 55664ff9bbee78a5a3cfdb59586f39783c937601, 2026-11-13..2026-11-22 inclusive; its log reports 14,400 rows per symbol. New patch directly continues at 2026-11-23 00:00 UTC. Existing files preserved; concatenate by open_time, deduplicate overlaps with older full histories and audit continuity before backtests. Prior compressed files not independently re-audited here.

New patch: 1,805 rows per symbol, total 16,245; all 9 files read back from commit cd16e425dcdc9b595ba53568c6dba73151651eb1 and checked for exact 60-second sequence, close time, 12 fields, OHLC bounds, nonnegative volume. No missing/duplicate minutes in new patch. BTC is reference, other 8 symbols are portfolio assets.

| Symbol | Previous patch | New patch | Rows |
|---|---|---|---:|
| XMRUSDT | [Nov13–22](XMRUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](XMRUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| DOGEUSDT | [Nov13–22](DOGEUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](DOGEUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| LTCUSDT | [Nov13–22](LTCUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](LTCUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| ETCUSDT | [Nov13–22](ETCUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](ETCUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| ADAUSDT | [Nov13–22](ADAUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](ADAUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| BTCDOMUSDT | [Nov13–22](BTCDOMUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](BTCDOMUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| 1000PEPEUSDT | [Nov13–22](1000PEPEUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](1000PEPEUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| BATUSDT | [Nov13–22](BATUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](BATUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |
| BTCUSDT | [Nov13–22](BTCUSDT_1m_UTC_20261113_THROUGH_20261122.csv.gz) | [Nov23–24](BTCUSDT_1m_UTC_20261123_THROUGH_20261124_0604.csv) | 1805 |

New CSV schema: open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base_volume,taker_buy_quote_volume,ignore. Raw Binance values retained. No interpolation, no forward fill, no spot substitution. Next continuation start: 2026-11-24T06:05:00Z. Portfolio metrics and trade exports not recalculated by this data update.
