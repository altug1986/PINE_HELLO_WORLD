# Monkey Test & Selection Bias — Quant Research Protocol (2026-11-17)

> **Status:** Methodology / research specification. No market-data backtest, TradingView reconciliation, Monte Carlo, DSR, or OOS validation has been executed for this document. Public, non-proprietary version of the CYBORG robustness research notes.

## Discovery

A strategy may finish positive even with negative expected value. A selected best window can look statistically extraordinary if compared with a randomly selected window instead of other best windows. A search over thousands of parameter combinations creates a further level of selection bias: **the entire selection procedure**, not just the winner, is the unit of inference.

### Reproducible synthetic example

- 500 independent trades; win payoff +2R, loss -1R, P(win)=1/3.
- Constant fee/slippage cost 0.05R per trade.
- Expected gross payoff = (1/3)*2R + (2/3)*(-1R) = 0.
- Expected net payoff = -0.05R/trade, or -25R over 500 trades.
- An observed sequence with 177 wins and 323 losses returns `177*2 - 323 - 500*0.05 = +6R`.
- Under the assumed binomial null, `P(Binomial(500, 1/3) >= 177) ≈ 0.1752632456`. A positive result is not, by itself, evidence of edge.

Numbers such as `p=0.0094` for selected-vs-random windows and `p=0.29` for best-vs-best windows may illustrate the selection problem but should **not** be treated as independently verified without the original data, code, metric and sampling procedure.

## Distinguish tests

| Method | Question | Limit |
|---|---|---|
| Monkey Entry | Is the entry timing better than a matched random timing policy? | Preserve cost and exposure; rerun exits on actual OHLC |
| Monkey Exit | Do exits improve over ex-ante specified alternative exits? | Stop/risk asymmetry can bias comparisons |
| Monkey Both | Does the combined strategy beat random policy families? | Null choice matters |
| Best-Window Monkey | Does the selected best window exceed **other strategies' own best windows**? | Apply the same selection across all metrics and window lengths |
| Block / trade bootstrap | How uncertain are a given strategy's outcomes? | Resampling the chosen trade distribution cannot prove edge |
| DSR | Could selected Sharpe arise from multiple trials and non-normal returns? | Report effective independent trials and dependencies |
| PBO / CSCV | Does the IS winner deteriorate out of sample across partitions? | Not a substitute for untouched chronological forward data |
| Walk-forward + held-out OOS | Does performance survive after parameters are locked? | Retuning on OOS invalidates independence |
| White Reality Check / Hansen SPA | Does a searched family contain genuine benchmark excess? | Requires a specified null, return matrix and appropriate bootstrap |

## Best-window matched-null protocol

1. Fix symbol, data coverage, timestamp conventions, order execution, fees, slippage, position sizing and risk controls.
2. Pre-register the statistic, benchmark and window set (e.g. best 40 consecutive trades, best 30/60/90 **calendar** days). Do not equate a 40-trade window to a 40-day window.
3. Compute real strategy statistic, e.g. `T_real=max_start net_PnL_30d(start)`.
4. Generate `B` matched random strategies and **fully resimulate** their fills and exits on the same price paths.
5. For each random strategy compute `T_b=max_start net_PnL_b_30d(start)`.
6. One-sided empirical p-value: `p=(1 + sum_b[T_b >= T_real])/(B+1)`.
7. If researchers also choose the *best* metric or best 30/60/90 duration after seeing results, reproduce that entire meta-selection inside each null draw or adjust the family of tests.
8. Report the whole distribution, not just the p-value: profit, Sharpe, Sortino, Calmar, drawdown percentiles, underwater duration, weekly negative-rate and stability measures.

**Caveat:** Matching trade count can remove frequency edge. Publish both frequency-matched and natural-frequency controls. For long-only trading, account for market drift with long-only random timing and constant-exposure benchmarks. Match exposure and market regimes where justified.

## Large-scale hyperparameter searches

After searching e.g. **35,000 highly correlated parameterizations**, comparing the winner to a single random strategy is not valid. Preserve the **full search log**, including rejected tests, code hashes, dataset versions, performance dates, seed, candidate relationships and all actual researcher choices.

Two distinct null hypotheses must not be conflated:

- **Timing null:** randomize signals while preserving a matched execution/risk framework; test entry/exit timing edge.
- **Family / data-snooping null:** compare the best searched candidate to the *maximally selected* benchmark-adjusted result under an appropriate dependence-preserving null (e.g. White/SPA family test).

A computationally defensible pipeline:
1. Align and audit candidate daily return vectors and benchmark.
2. Use common time-block indices across all candidate vectors in family-level bootstrap; maintain cross-strategy dependence.
3. Use true fill-level resimulation for shortlisted random-entry, random-exit and best-window trials.
4. Compute DSR/PBO with explicit assumptions and uncertainty.
5. Lock the winning configuration and evaluate on untouched chronological OOS.

Do not claim to have run 35,000 x 10,000 full simulations when only vectorized bootstrap or conceptual analysis was performed.

## Equity quality and acceptance

No single Sharpe threshold or p-value guarantees tradable edge. Evaluate:
- Net profit after all modeled costs, profit factor, Sharpe, Sortino, Calmar and Ulcer.
- Maximum drawdown; labeled P70/P80/P95/P99/P100 **with the distribution precisely specified**.
- Rolling underwater depth and duration, worst five days, negative weeks, and daily/weekly equity smoothness.
- CV = std(return)/mean(return) is unstable or undefined near zero/negative mean: never force a misleading favorable CV.
- Portfolio shocks require **synchronized time-series** and preserved inter-strategy correlation; the portfolio's risk is not the sum of isolated maximum drawdowns.

Simulation validity comes first: compare local fill timestamp, direction, size, price, commissions, slippage and PnL against the authoritative TradingView Strategy Tester export before claiming execution parity. No-repaint, confirmed bars, HTF closed snapshots and DCA behavior need explicit verification.

## Outputs / evidence ledger

Every report must label **Verified**, **Estimated**, **Unverified / Proposed**, or **Not tested**, including:
- OHLC symbol, venue, timezone, interval, missing bars, file SHA.
- Strategy version, parameter set, execution engine, code SHA.
- Costs, fill assumptions, sample dates and actual in-/out-of-sample split.
- Null generator, seed, number of simulations and confidence uncertainty.
- True selection history and whether OOS was touched during tuning.

**Core principle:** Validate execution → account for selection → inspect risk/linearity → verify untouched OOS. High in-sample returns alone are not proof of edge.

## Research background

- White (2000), *A Reality Check for Data Snooping*.
- Bailey et al., *The Probability of Backtest Overfitting*.
- Bailey & López de Prado, *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
- Hansen (2005), *A Test for Superior Predictive Ability*.

**Information policy:** This public note deliberately excludes proprietary Pine logic, terminal socket identifiers, OHLC datasets, real positions, portfolio allocations and private research exports. Full internal specification is retained in the private Drive CYBORG archive.
