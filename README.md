# Θ ThetaGang Θ

ThetaGang automates options portfolios at [Interactive Brokers](https://www.interactivebrokers.com/).
This repository holds two programs that share the same ideas but not their code:

| | `thetagang-nt` | `thetagang` |
|---|---|---|
| Package | [`thetagang_nt/`](thetagang_nt/) | [`thetagang/`](thetagang/) |
| Built on | [NautilusTrader](https://nautilustrader.io) 1.231 | [ib_async](https://github.com/ib-api-reloaded/ib_async) |
| Runs as | a long-running node: checks every few minutes | one pass per start: run it from a scheduler |
| Strategies | discretionary (rolls, closes, SPX-delta trades), VIX call hedge | wheel, discretionary, regime rebalancing, VIX call hedge, tail hedge, cash management |
| Monitoring | Prometheus metrics; Grafana dashboard and Telegram alerts in the separate `monitoring` repo | risk tables printed to the terminal |
| Config | [`thetagang.nt.toml`](thetagang.nt.toml) | [`thetagang.toml`](thetagang.toml), [`thetagang.discretionary.toml`](thetagang.discretionary.toml) |
| Python | 3.12+ (or Docker) | 3.11 – 3.14 |

## Risk disclaimer

**Options trading involves substantial risk.** Selling options can lose more
than the premium collected, and a bug or a bad setting can place real orders.
Run everything on a paper account first, keep the dry-run switches on until
you trust a setup, and understand every order the bot proposes.

---

## `thetagang-nt`: ThetaGang on NautilusTrader

### What it does

The node connects to TWS or IB Gateway through NautilusTrader's Interactive
Brokers adapter. On start it reconciles every position already in the account
(they appear as `EXTERNAL` positions), then runs two components side by side.

**PortfolioMonitor** ([`monitor.py`](thetagang_nt/monitor.py)) is a read-only
Nautilus `Actor`. Every `monitor.interval_secs` it quotes every position and
publishes Prometheus metrics on port 9464:

- account: net liquidation, available funds, margin used, unrealized P&L
- risk in SPX terms: beta-weighted delta (SPX units and $), delta $ without
  beta, gamma, theta per day, vega per vol point, and the number of positions
  without greeks
- per symbol: 1-year beta vs SPX, delta (shares and $), gamma, theta, vega
- per position: quantity, cost basis, price, unrealized P&L, P&L as a share of
  the premium, days to expiry, delta, implied volatility
- stress P&L for each SPX move in `monitor.stress_moves` (default ±1% (a typical day), ±5%, ±10%),
  with each stock moving by beta × the SPX move, and with every beta set to 1
- SPX, VIX and VIXMO levels, and open orders

Betas come from one year of IB daily bars, refreshed once a day, and are
shared with the strategy through the Nautilus cache. VIX options are left out
of the SPX numbers: they move against the market and price off VIX futures.

**Discretionary** ([`strategy.py`](thetagang_nt/strategy.py)) is a Nautilus
`Strategy`. Every `strategy.trading.interval_secs`, during NYSE hours, it works
through the account symbol by symbol:

1. **Short options and credit verticals** are rolled when
   `[strategy.credit.roll_when]` says so (near expiry with enough P&L, or a P&L
   target), or closed above `close_at_pnl`. A roll targets
   `[strategy.credit.roll_target]` (DTE, delta). The new strike may move against
   you by at most the premium already captured, and an in-the-money position
   only rolls to the same or a safer strike. Covered calls never roll below
   the share cost. When no roll is found, `close_if_unable_to_roll` closes a
   profitable position inside `max_dte` instead.
2. **Debit verticals and standalone long calls/puts** follow `[strategy.debit]`.
   A debit vertical is closed at `debit.take_profit` of its maximum profit; a
   long option once it is worth (1 + `debit.long_take_profit_gain`) × the
   premium paid. Otherwise both are rolled out once within
   `debit.roll_when.dte` days of expiry, or once the long leg's |delta| reaches
   `debit.roll_when.delta` (deep in the money), to `[strategy.debit.roll_target]` (the
   new long leg's DTE and delta), unless the roll costs more than
   `debit.max_roll_debit` × the width (a vertical) or × the old option's value
   (a long option).
3. **The VIX call hedge** sells all long VIX calls once VIX is above
   `close_hedges_when_vix_exceeds`. With no hedge held, it buys calls near
   `delta` and `target_dte` for `weight` × net liquidation, taking the
   weight from the first allocation band containing VIXMO.
4. **The trade plan** brings the account's SPX delta to 0 by resizing what
   you hold: each symbol moves toward its "SPX Δ=0 target" from the positions
   summary, by trading shares, one option, or a whole vertical as one combo,
   never flipping a position. A resize that would add short options (sell more
   of a short option or open credit spreads) isn't placed, since nothing checks
   margin for it; the plan shows it as "by hand". Index options are left alone.
   New positions are opened with [`pretrade`](#pre-trade-preview-trades-then-send-them).

   The plan waits
   when anything was rolled or closed in the same run, while the account is
   SPX delta neutral, and while any position has no greeks. An option with no
   quote at IB (an illiquid strike) has no greeks and so blocks the plan every
   day; set `ignore_missing_greeks = true` under `[strategy.symbols.<SYM>]` to
   count that symbol's unquoted options as zero delta instead. Neutral means
   |beta delta $| is within the band: a typical 1% SPX day may cost at most
   `delta_band.theta_days` days of theta, capped at `delta_band.nav_pct` of net
   liq. The dashboard colours SPX delta and beta delta $ green within the band,
   orange up to twice it, and red beyond. With `approve_trades = "manual"` the
   strategy only shows this plan; you open trades with
   [`pretrade`](#pre-trade-preview-trades-then-send-them).

Rolls and spread closes are sent as one IB combo order (a Nautilus
`OptionSpread`), priced at the net mid. An order still unfilled after
`reprice_after_secs` is moved halfway to the current mid, once. A symbol with
an open order at IB is skipped that run, whoever placed it (the strategy,
pretrade, or you in TWS: read over a read-only side connection, client id
`ib.client_id` + 100); when those orders can't be read, the run trades nothing.
The strategy leaves alone index options
(SPX, NDX, …), calendar and diagonal legs, and symbols with `no_trading`.

Contracts are chosen like this: load the chain for the first
`chain.expirations` expirations at or past the target DTE, quote the
`chain.strikes` strikes nearest the money, compute Nautilus greeks, and take
the nearest expiry, then the largest |delta| at or below the target.

> **`strategy.trading.dry_run = true` is the default.** Orders are logged and counted
> on the dashboard, never sent. Turn it off only after a paper run during
> market hours.

### Requirements

- TWS or IB Gateway with the API enabled (TWS: *Configure → API → Settings →
  Enable ActiveX and Socket Clients*). Default ports: TWS paper 7497, TWS
  live 7496, Gateway paper 4002, Gateway live 4001.
- **Docker Desktop** (recommended), or **Python 3.12+**. NautilusTrader 1.231
  doesn't install on 3.11, and the macOS wheels of 1.219–1.221 don't import
  on macOS 27.

### Quickstart with Docker

[`deploy/`](deploy/) runs thetagang-nt as two containers from one image:

| Container | Runs | Started by |
|---|---|---|
| `thetagang-monitor` | the portfolio monitor (`--role monitor`): dashboard metrics on `:9464`; no strategy, so it never trades; IB client `ib.client_id + 1` | `docker compose up -d`, always on |
| `thetagang-nt` | the trading strategy (`--role trade`): its own metrics (orders, dry run) on `:9465`; IB client `ib.client_id` | `docker compose --profile trading up -d`, only when you trade |
| `pretrade` | one-off and interactive: previews trades, sends them if you answer `y`; metrics on `:9466` while it asks; IB client `ib.client_id + 2` | `./pretrade.sh` (see [Pre-trade](#pre-trade-preview-trades-then-send-them)) |

The image holds only the dependencies; the repo is mounted read-only, so a
code or config change needs only a restart. Dashboards and alerts come from
the separate `monitoring` repo (see [Monitoring and alerts](#monitoring-and-alerts)).

1. Copy the sample config (`*.local.toml` files are git-ignored) and edit it:

   ```console
   cp thetagang.nt.toml thetagang.nt.local.toml
   ```

   Set at least `ib.port`, `ib.account_id` (or export `TWS_ACCOUNT`) and
   your `[strategy]` settings. Leave `ib.host = "127.0.0.1"`: in Docker the
   `IB_HOST` variable sends the bot to `host.docker.internal`.

2. Start the monitor, and the strategy when you want it trading:

   ```console
   cd deploy
   docker compose up -d --build                # the monitor only
   docker compose --profile trading up -d      # also the trading strategy
   docker compose logs -f thetagang-monitor    # or thetagang-nt
   ```

3. Check the metrics at <http://localhost:9464/metrics>, and the dashboard
   in Grafana (next section).

Day-to-day commands, run from `deploy/`:

```console
THETAGANG_CONFIG=thetagang.nt.toml docker compose up -d   # another config, path from the repo root
docker compose restart thetagang-monitor                  # after a code or config change (and thetagang-nt)
docker compose build                                      # after editing deploy/requirements.txt
docker compose stop thetagang-nt                          # stop trading, keep monitoring
docker compose --profile trading down                     # stop and remove both containers
```

Both containers restart by themselves after a crash or when Docker Desktop
starts, but stay down after `docker compose stop`.

### Pre-trade: preview trades, then send them

`pretrade` shows what trades would do to the portfolio before you send them.
It connects to IB, prices the trades at the live mid, and recomputes the whole
book (held positions + the proposal) with the same code as the dashboard. Then
it prints **Now / After / Change** for SPX delta, beta delta $, the neutral band
and neutrality, theta, vega, volga, gamma P&L, net carry, the break-even move
and the stress P&L, and asks before sending anything:

```console
cd deploy
./pretrade.sh                                        # your [[pretrade.candidates]], side by side (else the strategy's plan)
./pretrade.sh "sell 2 AA 20261218 40P"               # ad hoc: sell 2 AA Dec-18 40 puts
./pretrade.sh "buy 100 MS"                           # 100 shares
./pretrade.sh "buy 1 AA 20261218 40P/23P"            # a put debit spread: buy the 40 put, sell the 23 put
./pretrade.sh "sell 1 SPY 20261218 700/690P"         # a put credit spread (the right after the last strike covers both)
./pretrade.sh "sell 2 AA 20261218 40P" "buy 100 MS"  # several trades at once, judged together
```

```text
Proposed trades:
  MS                                     100 x   183.49 debit  $   +18,349
                                       +MS
  Total (+ cost, - credit)             $+18,349

                                   Now         After        Change
  SPX delta                     +12.12        +15.50         +3.38
  Beta delta $                $+93,005     $+118,965      $+25,960
  Neutrality               off neutral   off neutral
  ...
  Stress SPX -10%              $-4,034       $-6,630       $-2,596
Send (DRY RUN) these 1 order(s)? [y/N]
```

**Candidates.** List ideas to compare under `[[pretrade.candidates]]`, each a
name and one or more trades:

```toml
[[pretrade.candidates]]
name = "SPY put spread"
[[pretrade.candidates.trades]]
symbol = "SPY"
structure = "put_debit_spread"
width = 10.0
delta = 0.4
dte = 45
max_contracts = 5
```

Each candidate is sized **on its own** to bring SPX delta to 0 (its trades
together, split by `weight`). `./pretrade.sh` then shows every candidate's
orders and cost, and a table with **Now** and the portfolio after each
candidate: SPX delta, beta delta $, neutrality, theta, vega, volga, gamma P&L,
net carry, stress for each SPX move. It asks which one to send
(`1`, `2`, … or `0`/Enter for no trade). The candidates never change what the
strategy trades; it only resizes what you hold.

The **thetagang pre-trade** dashboard always shows the comparison: the trading
strategy re-sizes and compares the candidates every `refresh_mins` (15) during
market hours, publishing only (it never sends one). While `./pretrade.sh` runs,
the dashboard shows its fresher numbers instead. `refresh_mins = 0` under
`[pretrade]` turns the background comparison off.

Each ad-hoc trade is `buy|sell QTY SYMBOL [YYYYMMDD STRIKE(C|P)[/STRIKE(C|P)]]`.
QTY is shares for stock and contracts (or spreads) for options. A spread goes
out as one IB combo order: its first strike is bought on a `buy` and sold on a
`sell`, the second the other way. Without arguments it proposes your
candidates, or without any the strategy's trade plan (the resizes of what you
hold); sizing needs greeks for
every position, so run it during market hours.

On `y`, the trades are re-priced at the mid of that moment and sent through the
strategy's own order code, so `dry_run` still applies. Nothing is sent on any
other answer, or when you answer more than `proposal_max_age_secs` (5 min)
later. While it waits, the **thetagang pre-trade** dashboard in Grafana shows
the same numbers.

To approve every new trade yourself, set `approve_trades = "manual"` in
`[strategy.trading]`: the running strategy still rolls, closes and runs the VIX hedge,
but only shows its trade plan (status "awaiting approval") instead of opening
it.

Without Docker: `thetagang-nt -c thetagang.nt.local.toml pretrade "sell 2 AA 20261218 40P"`.

### Running without Docker

```console
python3.12 -m venv .venv-nt && source .venv-nt/bin/activate
pip install -e '.[nautilus]'
thetagang-nt -c thetagang.nt.local.toml --role monitor   # the portfolio monitor
thetagang-nt -c thetagang.nt.local.toml --role trade     # the strategy (another terminal)
```

`--role all` (the default) runs both in one process, with every metric on
`:9464`. Use an environment of its own: NautilusTrader upgrades pandas and
other packages that the `thetagang` environment relies on.

### Configuration

One TOML file, loaded into native Nautilus configs
([`config.py`](thetagang_nt/config.py)). [`thetagang.nt.toml`](thetagang.nt.toml)
lists every key with a comment.

| Section | Sets |
|---|---|
| `[node]` | trader id, log level, optional log directory, optional Redis cache |
| `[ib]` | host, port, client id, `account_id`, `market_data_type` (`REALTIME`, or `DELAYED_FROZEN` for last prices while markets are closed). Add `[ib.gateway]` to have Nautilus start IB Gateway in Docker |
| `[monitor]` | refresh interval, metrics address and port, stress moves |
| `[strategy.trading]` | `dry_run`, `interval_secs`, `market_hours_only`, `approve_trades` (`auto`/`manual`), `proposal_max_age_secs`, `reprice_after_secs`, `metrics_port`, `pretrade_metrics_port` |
| `[strategy.delta_band]` | `theta_days`, `nav_pct`: the SPX delta neutral band |
| `[strategy.chain]` | `strikes`, `expirations`, `quote_timeout_secs`, `minimum_credit`: picking a contract by delta |
| `[strategy.credit.roll_target]` | standalone short calls/puts and credit spreads: what a roll moves into (`dte`, `max_dte`, `delta`) |
| `[strategy.credit.roll_when]` | `pnl`, `dte`, `min_pnl`, `max_dte`, `close_at_pnl`, `close_if_unable_to_roll`; `.calls` and `.puts` set `itm`, `always_when_itm`, `credit_only` |
| `[strategy.debit]` | debit spreads and standalone long calls/puts: `take_profit` (spreads), `long_take_profit_gain` (longs), `max_roll_debit`; `.roll_when` (`dte`, `delta`, `max_dte`, `close_if_unable_to_roll`) and `.roll_target` (`dte`, `max_dte`, `delta` of the new long leg) |
| `[strategy.vix_call_hedge]` | `enabled`, `delta`, `target_dte`, `close_hedges_when_vix_exceeds`, `ignore_dte`, and `[[…allocation]]` bands (`lower_bound` ≤ VIXMO < `upper_bound`, first match wins, `weight` × net liquidation) |
| `[[pretrade.candidates]]` | `name`, and `[[pretrade.candidates.trades]]`: `symbol`, `structure` (`long_call`, `short_call`, `long_put`, `short_put`, `put_credit_spread`, `call_credit_spread`, `put_debit_spread`, `call_debit_spread`), `delta`, `dte`, `width` (spreads only), `weight`, `max_contracts`: ideas `pretrade` sizes and compares |
| `[strategy.butterfly]`, `[strategy.straddle]`, `[strategy.calendar]` | only with `[pm] decide = false`: what the bot's own rules would do for butterflies (`take_profit`, `partial_take_profit` of the max profit, `close_dte`), long straddles/strangles (`take_profit_gain`, `partial_take_profit_gain` of the debit, `leg_delta`, `roll_dte`) and calendars/diagonals (`take_profit` of the debit, `short_roll_dte`, `close_dte`). Each has `trade` (default false): true lets the bot place the recommendation (dry_run applies): close or close half as one combo with every leg in its ratio, close a straddle's winning leg, or roll a calendar's short leg to the next expiry before the long leg's for a net credit. "Roll whole" is never placed: it stays yours |
| `[[strategy.structures]]` | `name`, `legs` (contract labels as the dashboard shows them, e.g. `"AA 20261218 40P"`): option legs you hold as one position; recognised before any automatic grouping, left to you, and never resized by the trade plan |
| `[strategy.symbols.<SYM>]` | `no_trading`, `put_strike_limit`, `call_strike_limit`, `delta` and `max_dte` (override `credit.roll_target`), `ignore_missing_greeks` |
| `[pm]` | the Portfolio Manager: `decide` (default true: its review signals replace the bot's roll/close rules, which then only report), `journal`, `max_loss` (a risk budget in $ for every strategy: a P0 review when its worst case exceeds it; a strategy's own `max_loss` overrides it), and the review thresholds by vault register ID: `profit_levels`, `spread_max_profit` (T1), `credit_levels` (T2), `delta_review` (T4), `delta_drift` (T5), `gt_review`, `gt_horizon` (T6), `remaining_reward_risk` (T8), `near_expiry_days`, `near_money`, `stale_secs`, `wide_spread`, `wide_spread_min` (T9), `dte_windows` (T3), `gamma_tolerance`, `theta_tolerance` (T7), `grid_moves`, `iv_shifts` (M1), `vix_moves`, `vix_iv_shifts` (M11), `realized_vol_days` (M6), `utility_horizon`, `es_alpha`, `risk_aversion`, `capital_cost`, `roll_margin` (M7), `commission` (M9), `roll_scan`, `roll_quote_lines`, `roll_rescan_mins`, `roll_dte_multiples`, `roll_min_life`, `roll_min_extension`, `roll_strike_shifts`, `roll_strike_tolerance` (M13), `vix_roll_dte` (V1), `iv_model`, `iv_window`, `iv_min_samples`, `iv_view_shrink` (M8), `iv_term_days` (M3), `american_pricing`, `surface`, `surface_strikes` (M14); for the Short Options Manager: `capture_levels` (S1), `short_dte_bands` (S2), `loss_multiples` (S3), `assignment_days`, `ex_dividend_days` (S4), `tested_moves`, `tested_horizon` (S5); for long VIX options: `vix_profit_levels` (V2), `vix_loss_levels` (V3), `vix_review_dte` (V7). The Portfolio Manager dashboard's *Trading rules in use* table lists the trading rules (T1, T3–T6, T8, S1–S5, V1–V3, V7, `decide`), default or yours; model and data settings act but are not listed |
| `[[pm.strategies]]` | your inputs for one strategy, by its Portfolio Manager ID (e.g. `"AA-debit-spread-1"`): `thesis`, `invalidation_below` / `invalidation_above` (a price: the thesis is invalid past it, a P0 review), `target`, `max_loss` (your risk budget: a P0 review when the worst case exceeds it), `thesis_valid`, `assignment_intended` (short premium: expiry or assignment is the plan, so no assignment review) |

Remove the whole `[strategy]` section to run the monitor only. The roll,
discretionary and VIX keys mean the same as in `thetagang`, so values can be
copied across.

### Monitoring and alerts

Prometheus, Grafana and the Telegram alerts are a repo of their own,
`../monitoring`; this repo only serves metrics. Start it with `./run.sh up` in that repo. Its Prometheus scrapes this
bot on the host's `127.0.0.1:9464` (job `thetagang`) and keeps two years of
history; its Grafana (<http://localhost:3000>, user `admin`, password in that
repo's `.env`) shows the **thetagang** dashboard in its own **thetagang**
folder and sends the alerts to Telegram. To change them, edit
`grafana/dashboards/thetagang/thetagang.json` and the `thetagang` group in
`grafana/provisioning/alerting/rules.yaml` there.

Alerts are sent 09:30–16:00 New York time on weekdays, repeated at most every
4 hours while they last:

| Alert | Fires when |
|---|---|
| thetagang-nt down | the bot was running in the last 8 hours and its metrics are now unreachable (5 min) |
| data stale | no refresh for over 5 minutes (5 min) |
| positions without greeks | any position has no greeks for 30 min: SPX delta is incomplete and new trades are blocked |
| SPX delta out of band | \|beta delta $\| outside the neutral band for an hour with `dry_run = false` |
| stress loss | a −10% SPX move (beta-weighted) would lose more than 5% of net liquidation (15 min) |
| order sent | the bot sent a real (not dry-run) order in the last 10 minutes |
| PM review | a strategy has an open P0 or P1 Portfolio Manager review (not "data unreliable"): the message names the strategy, its signals and what the bot's rules say |

The dashboard (everything the old bot printed):

| Row | Panels |
|---|---|
| Headline | one tile each, the latest value (shown in any time range): net liquidation, unrealized P&L, SPX delta, theta per day, vega per vol point, volga (vega change per vol point), gamma, *attention* (strategies with an open P0 or P1 Portfolio Manager review; click for the review queue), data age |
| Account | available funds, cash, excess liquidity, initial and maintenance margin, buying power, cushion; SPX move today |
| Risk | portfolio greeks now / after the SPX Δ=0 proposals / after the trade plan; stress P&L, beta-weighted and with all betas at 1; the VIX call hedge (calls held, allocation at VIXMO, close level) and what it would do |
| By symbol | beta delta $ by symbol; today's P&L by symbol and its attribution (from SPX, unexplained, delta, gamma, theta, vega, residual); the Symbols table (beta, delta, delta $, beta delta $, gamma, theta, vega, volga) |
| Positions summary | shares, short/long puts and calls, average strikes, weight now and target weight for SPX Δ=0, target beta delta $, share-equivalent change, side, and the net target contracts (display only) |
| Structures | the Options section: each symbol's option legs grouped into the positions they form (your `[[strategy.structures]]` groups first, then butterflies, long straddles/strangles, verticals, calendars/diagonals, single legs), with combined cost, value, P&L, P&L %, max profit and loss and breakevens at expiry (one-expiry structures), Δ Γ Θ V and the underlying price; and the Portfolio Manager's strategy ID (click it for the strategy's review), priority and outcome |
| Positions | quantity, cost, price, unrealized P&L, P&L ratio, DTE |
| Trade plan | the configured trades sized to SPX Δ=0 and why (only while the trading strategy runs) |
| Activity | VIX and VIXMO, orders per hour (dry-run orders included) |

The monitor computes the actions and targets with the strategy's own rules
(`thetagang_nt/rules.py`) and the `[strategy]` settings, so they show what the
strategy would do even when it isn't running. Cash, margins, buying power,
cushion and IB's daily P&L come from a small read-only IB connection inside
the monitor (client id `ib.client_id + 101` in `--role monitor`), since
NautilusTrader doesn't expose them.

All metrics are named `thetagang_*`, so you can query them in Prometheus or
add your own panels.

### Portfolio Manager

Decision support per strategy, implementing the PM Knowledge Base vault
(`~/Documents/project/PM_Knowledge_Base`): it never trades. Where the vault and
the bot's own rules disagree, the vault wins: with `[pm] decide = true` (the
default) the bot's roll/close rules only report, and the Portfolio Manager's
**review signals** tell you what to look at. You decide; the trade plan and
the VIX hedge are unchanged.

Every monitor cycle, for each strategy (the structures above, with a stable
ID such as `AA-debit-spread-1` that survives rolls), the pure functions in
[`thetagang_nt/pm/`](thetagang_nt/pm) compute:

- **Data quality**: missing, crossed or stale quotes, a long leg with no bid,
  no greeks. A wide spread is a real market, not bad data: a P2 signal with its
  cost to cross, and the leg's model values are rough. A few held options per
  cycle are read from IB directly, because the Nautilus adapter drops a side IB
  marks as none and a quiet quote looks stale. Also: **executable value** (sell at the bid, buy back at the ask) next to
  the mid, and the cost to close now.
- **Convexity class** from the net Gamma and Theta (long, short, mixed), never
  from how the strategy was opened; mixed for several expiries or when gamma
  changes sign within ±5%.
- **Gamma/Theta efficiency** for long convexity: GT, GT_h over 1/5/10 trading
  days at the 21-day realised volatility, and the breakeven volatility.
- **Profit**: return on debit or credit captured, remaining reward/risk.
- **Scenario grid** ([`pm/core/scenarios.py`](thetagang_nt/pm/core/scenarios.py)):
  every leg repriced from a model baseline, not the mid. Equity and ETF options
  are American with discrete dividends
  ([`pm/core/american.py`](thetagang_nt/pm/core/american.py): Barone-Adesi and
  Whaley for puts, Black's pseudo-American for calls, checked against a binomial
  tree in the tests; IV re-implied by the same model). IB gives one dated dividend
  and a 12-month total, so later dividends are estimates. Until an underlying's
  dividends load, and for index options, it is Black-Scholes; VIX options are
  Black-76 on their future. Cells are after
  1, 5 and 10 trading days, at price moves of ±2 / ±5% and ±2 / ±1 realised
  moves, and IV shifts of ±5 / ±10 points (VIX: futures −20…+30%, IV ±10 / ±20).
  Legs expiring inside the horizon go to intrinsic, with their strikes as extra
  price nodes. Named stresses beyond the grid: gap down / up (±10%),
  event IV crush and front-month spike (VIX: spike, fade, regime shift). The
  worst grid loss (WCL) and the worst stress loss are the P0 risk-budget
  checks against `max_loss + min(0, unrealised P&L)`.
- **Forward value** ([`pm/core/valuation.py`](thetagang_nt/pm/core/valuation.py),
  Phase 2): expected P&L and Expected Shortfall over `utility_horizon` days,
  with the realised-vol forecast, zero drift and IV unchanged as the
  probabilities, and the HOLD and CLOSE utilities of the vault's roll_rules.
  Reported only, not a signal: with zero drift and IV unchanged, HOLD scores
  below CLOSE almost by construction. No probabilities (and no forward value)
  without a realised-vol forecast, e.g. for VIX options.
- **Small IV surface** ([`pm/core/surface.py`](thetagang_nt/pm/core/surface.py)):
  a smile per held expiry from about 5 out-of-the-money strikes around the money,
  one expiry per cycle, quoted and unsubscribed like the roll scan. The grid stays
  sticky strike; the dashboard shows how its 5-day P&L changes if IV stays with
  moneyness instead. Model risk, not a signal. VIX and index options are left out.
- **Joint price and IV model**
  ([`pm/core/iv_model.py`](thetagang_nt/pm/core/iv_model.py)), reported beside
  that one: each underlying's two years of daily IV and price bars from IB, the
  5-day moves from days with IV in the same third of its trailing year as
  today's. Returns are demeaned and scaled to the realised-vol forecast (no view
  on price); the expected IV change is halved, since the evidence for IV mean
  reversion is thin; longer-dated legs move by `sqrt(30 / DTE)` of it. One
  underlying loads per cycle, so a restart takes about half an hour to fill.
  VIX options are left out (they need the vault's M12).
- **Review signals**, by the strategy's manager: how it was opened, not its
  convexity class (a debit spread can be short gamma). Both share P0 hard
  (thesis invalid, risk budget breached, data unreliable) and P2 class change.
  - *Long Options Manager* (opened for a debit,
    [`pm/long_options/`](thetagang_nt/pm/long_options)): P1 profit levels on
    the debit, the DTE window for the strategy's original life, a short leg
    near expiry and the money; P2 deep-ITM long option, GT_h below 1, low
    remaining reward. Long VIX options
    ([`pm/long_options/vix.py`](thetagang_nt/pm/long_options/vix.py), vault
    `vix_options_management`) are priced on their matching VIX future (put-call
    parity at the held strike) and use VIX profit levels (10 / 15 / 20 / 30% on
    the debit, V2), loss reviews (-25 / -35%, V3) and a fixed time review (14
    days left, V7) in place of the general profit levels and DTE window.
  - *Short Options Manager* (opened for a credit,
    [`pm/short_options/`](thetagang_nt/pm/short_options)): P1 premium captured
    (50 / 70 / 80 / 90%), the front short leg's DTE band (21 / 14 / 7 days),
    the loss as a multiple of the credit, an in-the-money short leg near
    expiry (assignment), and, in either manager, an in-the-money short call
    near an ex-date with extrinsic value below the dividend; P2 spot within one 5-day move of a short strike. It
    also shows the loss multiple and the premium left per $ of risk.

  The outcome is HOLD, REVIEW, REVIEW - data unreliable, or CLOSE review (any
  other P0). Shared engines live in [`pm/core/`](thetagang_nt/pm/core).

**Roll candidates** ([`pm/core/rolls.py`](thetagang_nt/pm/core/rolls.py)): for
a strategy not on HOLD, the same structure at later expiries (back to its
original life and longer; VIX 30-60 days) and shifted strikes, each priced at
executable quotes and judged as a new trade through the same grid and utility.
The dashboard ranks them and shows the model's HOLD / CLOSE / ROLL view. This
is reported, never a signal or an order: under zero drift and unchanged IV a
new trade rarely beats cash after its costs. IB limits market data lines, so
the monitor scans one strategy per cycle after publishing, quotes a few
contracts at a time and unsubscribes them (`roll_quote_lines`,
`roll_rescan_mins`; `roll_scan = false` turns it off).

Every threshold is a hypothesis with its vault register ID, set under `[pm]`.
Left of phase 2: a fuller surface (a real fit, arbitrage checks, every expiry).

The **journal** (Postgres, the `journal-db` service) keeps the strategies and
their legs (opened and closed), one snapshot per strategy per day, a
decision-log row whenever a strategy's outcome or signals change, and each held
option's implied vol per day (`iv_history`, legs with a data flag skipped): the
history an IV-change distribution will need. It is also the **forward test**
(`forecasts`, `leg_marks`, `roll_forecasts`): what the model said each day, the
day's last good quotes, and bid and ask on every leg, kept before the outcome is
known. The `forecast_results` view compares each forecast with the same legs 5
recorded days later; the Portfolio Manager dashboard shows it by view. IB has no
daily or expired option bars, so a real backtest needs bought data (Phase 3). Positions
held before the journal started have an estimated entry date. Grafana reads
it with a read-only login. Both passwords live in `deploy/.env` (git-ignored):

```bash
# deploy/.env
JOURNAL_DB_PASSWORD=...        # the journal database
JOURNAL_GRAFANA_PASSWORD=...   # Grafana's read-only login; the same line goes in ../monitoring/.env
```

The monitor creates the database tables and Grafana's login (`grafana_ro`,
SELECT only) itself. A database problem is logged and skipped; the dashboard
never waits on it.

The **Portfolio Manager** dashboard (linked from the thetagang dashboard)
shows the review queue (every strategy not on HOLD, what you can act on
first: real P0, then P1, then P2, each with clean prices before flagged
ones, strategies with only bad data last; signals one line per priority;
with its manager, next to what the bot's rules say), each strategy's detail
(values, class, efficiency, for short options the credit captured, loss
multiple and premium left per $ of risk, signals, your thesis and budget, P&L
history, decision log, legs), the book's class and manager mix and
concentration, and data quality.

### Differences from `thetagang`'s discretionary strategy

- **Greeks** come from Nautilus' Black-Scholes model on each option's mid, not
  from IB's model greeks, so deltas differ slightly from TWS. VIX options are
  priced on the VIX forward implied by put-call parity.
- **No open-interest filter** when choosing contracts: the Nautilus IB adapter
  doesn't provide open interest.
- **VIX allocation bands** match strictly `lower_bound` ≤ VIXMO <
  `upper_bound`, and a hedge close sells only long VIX calls.
- **The trade plan only resizes what you hold** (the old `place_target_orders`);
  there is no `[[strategy.trades]]`: open new positions with `pretrade`.
  No orders are cancelled at startup;
  the open-order check per symbol prevents duplicates instead.

### Troubleshooting

- **The bot can't connect from Docker.** Check the API is enabled on the
  configured port, and that TWS accepts the connection (*Trusted IPs*, or
  allow connections from outside localhost).
- **The dashboard shows no data.** Is the monitor running (`docker compose ps`)
  and is the monitoring stack up (`./run.sh status` in `../monitoring`)? Prometheus' target page
  (<http://localhost:9090/targets>) shows the `thetagang` job's state.
- **"No market data during competing live session".** The same IB user is
  logged in elsewhere, for example live TWS while testing paper.
- **Many "no underlying price" warnings.** The market is closed or quotes are
  missing; those positions are skipped and new trades wait. Use
  `market_data_type = "DELAYED_FROZEN"` to see last prices on weekends.
- **"Order overfill rejected" at startup.** Nautilus can't reconcile
  fractional shares. It doesn't affect the strategy.
- **The other bot runs out of market data.** Both share IB's per-account limit
  on simultaneous quotes. Lower `chain.strikes` / `chain.expirations`, or
  don't run both during market hours.

---

## `thetagang`: the original bot

### How a run works

Each start is one pass ([`thetagang.py`](thetagang/thetagang.py)):

1. Load the config, migrating a v1 file to schema v2 first if needed, and
   expand environment variables (`$VAR`, `${VAR}`, `${VAR:-default}`; an
   unset variable without a default stops the run).
2. Check exchange hours (`[runtime.exchange_hours]`, NYSE by default). When
   the exchange is closed, or inside `delay_after_open` / `delay_before_close`,
   `action_when_closed` decides: `exit` (default), `wait` (up to
   `max_wait_until_open`) or `continue`.
3. Connect to IB, through [IBC](https://github.com/IbcAlpha/IBC) and its
   watchdog, or directly to a TWS/Gateway you started yourself.
4. Run the enabled stages once, in a fixed order, then disconnect and exit.

Run it from cron, launchd or a loop to repeat it during the day.

> **On this branch `thetagang` always runs as a dry run against a TWS or IB
> Gateway you start yourself.** [`main.py`](thetagang/main.py) passes
> `without_ibc=True, dry_run=True` whatever flags are given, so it shows
> orders without sending them. Edit that call to trade or to use IBC.

### Strategies

`run.strategies` picks what runs; each strategy enables a set of stages:

| Strategy | Stages | What it does |
|---|---|---|
| `wheel` | write puts, write calls, buy/sell shares, roll, close | the modified wheel on the weighted symbols in `[portfolio.symbols]` |
| `discretionary` | roll, close, discretionary | manages every option already held: rolls/closes short options and credit verticals, manages debit verticals, prints SPX beta-weighted greeks and stress P&L, sizes `[[strategies.discretionary.trades]]` to SPX delta 0 |
| `regime_rebalance` | regime share rebalancing | trades the shares of `strategies.regime_rebalance.symbols` back toward their target weights once they drift past the soft or hard band, only when the market is choppy (choppiness and efficiency gates) and outside a cooldown |
| `vix_call_hedge` | VIX call hedge | buys VIX calls by VIXMO allocation, sells them all above a VIX level |
| `tail_hedge` | tail-hedge options | keeps a ladder of long out-of-the-money puts, bought in tranches through the year within an annual budget |
| `cash_management` | cash management | buys a cash fund (e.g. SGOV) when cash is above target, and sells it when cash drops below |

`discretionary` can't be combined with `wheel` or `regime_rebalance`, which
size positions by portfolio weight. Stages always run in this order: write
puts, write calls, regime rebalance, buy shares, sell shares, roll, close,
discretionary, VIX hedge, tail hedge, cash management.

### Configuration (schema v2)

| Section | Sets |
|---|---|
| `[meta]` | `schema_version = 2` |
| `[run]` | `strategies` (or explicit `stages`) |
| `[runtime.account]` | account number, `cancel_orders` at startup, `margin_usage` (wheel sizing) |
| `[runtime.option_chains]` | how many expirations and strikes to scan |
| `[runtime.orders]` | exchange, `minimum_credit`, price adjustment delay, algo, estimated fees |
| `[runtime.exchange_hours]` | exchange and the closed-market behaviour above |
| `[runtime.database]` | state database (SQLite at `data/thetagang.db` next to the config by default; or a `url`) |
| `[runtime.ib_async]`, `[runtime.ibc]`, `[runtime.watchdog]` | connection, IBC login and gateway watchdog |
| `[portfolio.symbols.<SYM>]` | wheel weights and per-symbol settings; for `discretionary`, optional overrides only |
| `[strategies.wheel]` | `defaults.target`, `defaults.write_when`, `defaults.roll_when` (also used by `discretionary` for rolls), constants, rebalancing |
| `[strategies.discretionary]` | `delta_band`, `trades`, `place_target_orders`, `adjust_price_after_delay`, debit-spread settings, `long_option_alert_delta` |
| `[strategies.vix_call_hedge]`, `[strategies.tail_hedge]`, `[strategies.cash_management]`, `[strategies.regime_rebalance]` | the matching strategies |

The samples explain every key: [`thetagang.toml`](thetagang.toml) (wheel +
VIX hedge + cash management) and
[`thetagang.discretionary.toml`](thetagang.discretionary.toml) (discretionary).

### Usage

```console
uv sync                                      # or: pip install -e .
thetagang -c thetagang.local.toml            # one pass
thetagang -c thetagang.toml --migrate-config # migrate a v1 config to v2 and exit
thetagang -h
```

| Flag | Effect |
|---|---|
| `-c`, `--config` | config file (default `thetagang.toml`) |
| `--dry-run` | show orders without sending them (always on in this branch) |
| `--without-ibc` | connect to a TWS/Gateway you run yourself (always on in this branch) |
| `--migrate-config`, `--yes` | migrate a v1 config to v2 and exit; `--yes` skips the prompt |
| `-v`, `--verbosity` | log level |

Every option can also be set as `THETAGANG_<OPTION>` in the environment.

### Further reading

- [`PORTFOLIO_RISK_GUIDE.md`](PORTFOLIO_RISK_GUIDE.md): reading the
  discretionary strategy's greeks, stress and delta-by-symbol tables
- [`docs/tail-hedging.md`](docs/tail-hedging.md): the tail hedge
- [`docs/external-decisions.md`](docs/external-decisions.md) and
  [`examples/external_decisions/`](examples/external_decisions/): external
  decision providers for regime target weights and tail-hedge harvesting
- [`docs/accounting.md`](docs/accounting.md): how account values and pending
  order cash are computed

---

## Development

```console
uv sync                       # thetagang and its dev tools (Python 3.11+)
uv run pytest                 # thetagang tests (tests/)
uv run ruff check . && uv run ruff format .

pip install -e '.[nautilus]' pytest   # in a Python 3.12+ environment
pytest tests_nt                        # thetagang-nt tests
```

| Path | Contents |
|---|---|
| `thetagang_nt/rules.py` | pure decision rules: spread pairing, roll and close rules, strike limits, contract choice, trade sizing, VIX bands, beta |
| `thetagang_nt/portfolio.py` | Nautilus cache → positions, prices, greeks, SPX book, stress P&L |
| `thetagang_nt/monitor.py`, `strategy.py` | the Actor and the Strategy |
| `thetagang_nt/aio.py` | awaiting Nautilus requests from coroutines |
| `thetagang_nt/config.py`, `node.py` | TOML → Nautilus configs; the `thetagang-nt` entry point |
| `deploy/` | Dockerfile and docker-compose for the bot |
| `thetagang/` | the original bot |
| `tests/`, `tests_nt/` | tests for each |

## License

AGPL-3.0-only. See [`LICENSE`](LICENSE).
