# Offline TradingView-Style Rust Research Engine (MVP)

**Status: research prototype. NOT a general Pine v6 interpreter and NOT verified to match TradingView Deep Backtesting.** This is synthetic, generic code only. The original public README and development protocol remain untouched on main.

## What is supported

Deterministic one-pass OHLC replay; signals at bar close with market execution at next open by default, or at close with `--on-close`; LONG/SHORT, ADD (max legs), REDUCE fraction, EXIT; fixed cash per leg, two-sided commission and adverse tick slippage. Emits trades, marked equity, closed PnL, PF, close DD and intrabar adverse proxy.

## Run

```bash
cd rust-backtester
cargo test
cargo run --release -- --ohlc data/ohlc.csv --signals data/signals.csv --out results/demo --initial 2500 --cash-per-leg 250 --commission-percent 0.04 --slippage-ticks 1 --tick-size 0.001 --max-legs 2 --on-close
```

Inputs: OHLC CSV `timestamp,open,high,low,close`; signals CSV `timestamp,action,fraction`. Time strings must match exactly. Normalized plain CSV only. `REDUCE,0.10` closes 10% quantity. Open positions aren't forcibly closed at dataset end. When a LONG signal arrives in a short, it only closes the short (no instant reverse).

**Not implemented:** Direct arbitrary Pine upload, strategy.exit stop/limit/trailing semantics, MTF request.security snapshots, bar magnifier, broker's precise intrabar ordering, margin/funding, rounding/currency conversion, lot-level bookkeeping, deep-backtest EMA initialization, TV Sharpe/Sortino/Calmar parity or OOS/Monte Carlo. Never infer parity from similar net profit. Full offline starter ZIP in the chat contains a richer research implementation and a narrow Python EMA Pine template bridge; this GitHub branch currently contains the minimal Rust kernel.

## Validation discipline

Export a genuine TradingView Deep Backtesting List of Trades; normalize and compare **every entry/exit timestamp, direction, quantity, price and fee**. Treat the first mismatch as a failing parity gate. No proprietary Pine scripts, OHLC datasets, user trades or credentials belong in this public repository.
