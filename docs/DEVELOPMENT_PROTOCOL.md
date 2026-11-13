# Systematic Trading — Development & Validation Protocol

> This is a **sanitized process document**, adapted from the connected Google Drive working protocol. It contains **no proprietary Pine strategies**, trading signals, account information, or portfolio results.
>
> Status: **process guidance only**; not a strategy, backtest, or TradingView validation.

## 1. Source of truth

- Keep a clearly named, immutable **reference/baseline** version of each strategy.
- Develop changes in a separate candidate version or branch.
- Record the date, symbol, timeframe, data range, fees, slippage, and input parameters of each test.
- Prefer the most recently **verified** reference to a merely newer candidate.
- Never silently overwrite the baseline or confuse research variants with live versions.

## 2. Pine Script execution consistency

- Explicitly document whether orders are processed on bar close or intrabar.
- For higher-timeframe data, use deliberate snapshot timing and guard against lookahead/repaint.
- Keep entry, exit, risk, and alert logic auditable as separate sections.
- Preserve existing alert payloads, DCA constraints, and state transitions unless the change explicitly targets them.
- Test the **flat → entry → exit → re-entry** lifecycle and same-bar reversal edge cases.

## 3. Backtest validation gate

A candidate is **not validated** merely because it compiles or an offline engine returns attractive metrics.

1. Identify a reference TradingView Strategy Tester export.
2. Match historical bars, timezone and contract identifiers, cost model, position sizing, and order timing.
3. Compare trade list, direction, entry/exit timestamps, quantities, fees, and P&L with the reference.
4. Investigate mismatches before reporting exact equivalence.
5. Evaluate in-sample and out-of-sample periods separately, with walk-forward and robustness checks when applicable.
6. Monitor drawdown, underwater duration, return concentration, Profit Factor, Sharpe, Sortino, Calmar, Ulcer, weekly consistency, and dispersion.

## 4. Change-control checklist

- [ ] Baseline source preserved
- [ ] Modification isolated and documented
- [ ] Pine syntax / static validation checked
- [ ] Regression checks on entries, exits, DCA, alarms, and state performed
- [ ] TradingView reference matched, or mismatch explicitly disclosed
- [ ] Out-of-sample evaluation completed
- [ ] Status marked: proposed / research / candidate / verified / retired

## 5. Repository boundary

**This repository is public.** Do not place proprietary Pine source, private research notes, account details, API keys, Webhook secrets, live position information, or raw account-level exports here.

Store proprietary sources and authenticated test data only in a **private** repository with narrowly scoped access.

---
Source: Google Drive, *CYBORG DESTEK — Günlük 11:45 Çalışma Protokolü* (sanitized and condensed).