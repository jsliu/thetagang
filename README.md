# Θ ThetaGang Θ

ThetaGang automates options portfolios at [Interactive Brokers](https://www.interactivebrokers.com/).
This repository holds two programs that share the same ideas but not their code:

| | `thetagang-nt` | `thetagang` |
|---|---|---|
| Package | [`thetagang_nt/`](thetagang_nt/) | [`thetagang/`](thetagang/) |
| Built on | [NautilusTrader](https://nautilustrader.io) 1.231 | [ib_async](https://github.com/ib-api-reloaded/ib_async) |
| Runs as | a long-running node: checks every few minutes | one pass per start: run it from a scheduler |
| Strategies | discretionary (rolls, closes, SPX-delta trades), VIX call hedge | wheel, discretionary, regime rebalancing, VIX call hedge, tail hedge, cash management |
| Monitoring | Prometheus metrics, Grafana dashboard and Telegram alerts (in the intraday project's monitoring stack) | risk tables printed to the terminal |
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
- stress P&L for each SPX move in `monitor.stress_moves` (default ±5%, ±10%),
  with each stock moving by beta × the SPX move, and with every beta set to 1
- SPX, VIX and VIXMO levels, open orders, and long options whose |delta| has
  reached `long_option_alert_delta`

Betas come from one year of IB daily bars, refreshed once a day, and are
shared with the strategy through the Nautilus cache. VIX options are left out
of the SPX numbers: they move against the market and price off VIX futures.

**Discretionary** ([`strategy.py`](thetagang_nt/strategy.py)) is a Nautilus
`Strategy`. Every `strategy.interval_secs`, during NYSE hours, it works
through the account symbol by symbol:

1. **Short options and credit verticals** are rolled when
   `[strategy.roll_when]` says so (near expiry with enough P&L, or a P&L
   target), or closed above `close_at_pnl`. A roll targets
   `[strategy.roll_target]` (DTE, delta). The new strike may move against
   you by at most the premium already captured, and an in-the-money position
   only rolls to the same or a safer strike. Covered calls never roll below
   the share cost. When no roll is found, `close_if_unable_to_roll` closes a
   profitable position inside `max_dte` instead.
2. **Debit verticals** are closed at `debit_spread_take_profit` of their
   maximum profit, and rolled out once within `roll_when.dte` days of expiry,
   unless the roll costs more than `debit_spread_max_roll_debit` × the width.
3. **The VIX call hedge** sells all long VIX calls once VIX is above
   `close_hedges_when_vix_exceeds`. With no hedge held, it buys calls near
   `delta` and `target_dte` for `weight` × net liquidation, taking the
   weight from the first allocation band containing VIXMO.
4. **New trades** in `[[strategy.trades]]` are sized together so the
   account's SPX delta goes to 0, split by `weight`, in whole contracts,
   never in the wrong direction, and capped by `max_contracts`. They wait
   when anything was rolled or closed in the same run, while |SPX delta| is
   within `delta_band`, and while any position has no greeks.

Rolls and spread closes are sent as one IB combo order (a Nautilus
`OptionSpread`), priced at the net mid. An order still unfilled after
`reprice_after_secs` is moved halfway to the current mid, once. A symbol with
an open order is skipped that run. The strategy leaves alone index options
(SPX, NDX, …), calendar and diagonal legs, and symbols with `no_trading`.

Contracts are chosen like this: load the chain for the first
`chain_expirations` expirations at or past the target DTE, quote the
`chain_strikes` strikes nearest the money, compute Nautilus greeks, and take
the nearest expiry, then the largest |delta| at or below the target.

> **`strategy.dry_run = true` is the default.** Orders are logged and counted
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

[`deploy/`](deploy/) runs the bot. The image holds only the dependencies; the
repo is mounted read-only, so a code or config change needs only a restart.
Dashboards and alerts come from the intraday project's monitoring stack (see
[Monitoring and alerts](#monitoring-and-alerts)).

1. Copy the sample config (`*.local.toml` files are git-ignored) and edit it:

   ```console
   cp thetagang.nt.toml thetagang.nt.local.toml
   ```

   Set at least `ib.port`, `ib.account_id` (or export `TWS_ACCOUNT`) and
   your `[strategy]` settings. Leave `ib.host = "127.0.0.1"`: in Docker the
   `IB_HOST` variable sends the bot to `host.docker.internal`.

2. Start the bot:

   ```console
   cd deploy
   docker compose up -d --build
   docker compose logs -f thetagang-nt
   ```

3. Check its metrics at <http://localhost:9464/metrics>, and the dashboard
   in Grafana (next section).

Day-to-day commands, run from `deploy/`:

```console
THETAGANG_CONFIG=thetagang.nt.toml docker compose up -d   # another config, path from the repo root
docker compose restart thetagang-nt                       # after a code or config change
docker compose build thetagang-nt                         # after editing deploy/requirements.txt
docker compose stop thetagang-nt                          # stop the bot
docker compose down                                       # stop and remove the container
```

The bot container restarts by itself after a crash, but stays down after
`docker compose stop`.

### Running without Docker

```console
python3.12 -m venv .venv-nt && source .venv-nt/bin/activate
pip install -e '.[nautilus]'
thetagang-nt -c thetagang.nt.local.toml
```

Use an environment of its own: NautilusTrader upgrades pandas and other
packages that the `thetagang` environment relies on. Monitoring works the
same: the bot serves `:9464` on the host either way.

### Configuration

One TOML file, loaded into native Nautilus configs
([`config.py`](thetagang_nt/config.py)). [`thetagang.nt.toml`](thetagang.nt.toml)
lists every key with a comment.

| Section | Sets |
|---|---|
| `[node]` | trader id, log level, optional log directory, optional Redis cache |
| `[ib]` | host, port, client id, `account_id`, `market_data_type` (`REALTIME`, or `DELAYED_FROZEN` for last prices while markets are closed). Add `[ib.gateway]` to have Nautilus start IB Gateway in Docker |
| `[monitor]` | refresh interval, metrics address and port, long-option alert delta, stress moves |
| `[strategy]` | `dry_run`, run interval, `market_hours_only`, `delta_band`, `minimum_credit`, debit-spread take-profit and maximum roll debit, `reprice_after_secs`, `chain_strikes`, `chain_expirations`, quote timeout |
| `[strategy.roll_target]` | what a roll moves into: `dte`, `max_dte`, `delta` |
| `[strategy.roll_when]` | `pnl`, `dte`, `min_pnl`, `max_dte`, `close_at_pnl`, `close_if_unable_to_roll`; `.calls` and `.puts` set `itm`, `always_when_itm`, `credit_only` |
| `[[strategy.trades]]` | `symbol`, `structure` (`long_call`, `short_call`, `long_put`, `short_put`, `put_credit_spread`, `call_credit_spread`, `put_debit_spread`, `call_debit_spread`), `delta`, `dte`, `width` (spreads only), `weight`, `max_contracts` |
| `[strategy.vix_call_hedge]` | `enabled`, `delta`, `target_dte`, `close_hedges_when_vix_exceeds`, `ignore_dte`, and `[[…allocation]]` bands (`lower_bound` ≤ VIXMO < `upper_bound`, first match wins, `weight` × net liquidation) |
| `[strategy.symbols.<SYM>]` | `no_trading`, `put_strike_limit`, `call_strike_limit`, `delta`, `max_dte` |

Remove the whole `[strategy]` section to run the monitor only. The roll,
discretionary and VIX keys mean the same as in `thetagang`, so values can be
copied across.

### Monitoring and alerts

There is one Prometheus + Grafana for both of your projects: the intraday
project's stack (`../intraday/monitoring`, started with
`./intraday/run_monitoring.sh up` from the intraday repo). Its Prometheus scrapes
this bot on the host's `127.0.0.1:9464` (job `thetagang`), keeps two years of
history, and its Grafana (<http://localhost:3000>, user `admin`, password in
intraday's `monitoring/.env`) shows the **thetagang** dashboard next to
intraday's and sends alerts to Telegram. The files live in that repo:
`monitoring/grafana/dashboards/thetagang.json` and the `thetagang` group in
`monitoring/grafana/provisioning/alerting/rules.yaml`.

Alerts are sent 09:30–16:00 New York time on weekdays, repeated at most every
4 hours while they last:

| Alert | Fires when |
|---|---|
| thetagang-nt down | the bot was running in the last 8 hours and its metrics are now unreachable (5 min) |
| data stale | no refresh for over 5 minutes (5 min) |
| positions without greeks | any position has no greeks for 30 min: SPX delta is incomplete and new trades are blocked |
| long option to roll by hand | a long option's \|delta\| reached `long_option_alert_delta` |
| SPX delta out of band | \|SPX delta\| > `delta_band` for an hour with `dry_run = false` |
| stress loss | a −10% SPX move (beta-weighted) would lose more than 5% of net liquidation (15 min) |
| order sent | the bot sent a real (not dry-run) order in the last 10 minutes |

The dashboard:

| Row | Panels |
|---|---|
| Headline | net liquidation, unrealized P&L, SPX delta, theta per day, vega per vol point, gamma, *need attention* (positions without greeks + long options at the alert delta), data age |
| History | net liquidation, SPX delta |
| Risk | beta delta $ by symbol; stress P&L, beta-weighted and with all betas at 1 |
| Detail | per-symbol table (beta, delta, delta $, beta delta $, theta, vega); per-position table (quantity, cost, price, unrealized P&L, P&L ratio, DTE) |
| Activity | unrealized P&L, VIX and VIXMO, orders per hour (dry-run orders included) |

All metrics are named `thetagang_*`, so you can query them in Prometheus or
add your own panels.

### Differences from `thetagang`'s discretionary strategy

- **Greeks** come from Nautilus' Black-Scholes model on each option's mid, not
  from IB's model greeks, so deltas differ slightly from TWS. VIX options are
  priced on the VIX forward implied by put-call parity.
- **No open-interest filter** when choosing contracts: the Nautilus IB adapter
  doesn't provide open interest.
- **VIX allocation bands** match strictly `lower_bound` ≤ VIXMO <
  `upper_bound`, and a hedge close sells only long VIX calls.
- **Not ported:** `place_target_orders`. No orders are cancelled at startup;
  the open-order check per symbol prevents duplicates instead.

### Troubleshooting

- **The bot can't connect from Docker.** Check the API is enabled on the
  configured port, and that TWS accepts the connection (*Trusted IPs*, or
  allow connections from outside localhost).
- **The dashboard shows no data.** Is the bot running (`docker compose ps`)
  and is the intraday monitoring stack up? Prometheus' target page
  (<http://localhost:9090/targets>) shows the `thetagang` job's state.
- **"No market data during competing live session".** The same IB user is
  logged in elsewhere, for example live TWS while testing paper.
- **Many "no underlying price" warnings.** The market is closed or quotes are
  missing; those positions are skipped and new trades wait. Use
  `market_data_type = "DELAYED_FROZEN"` to see last prices on weekends.
- **"Order overfill rejected" at startup.** Nautilus can't reconcile
  fractional shares. It doesn't affect the strategy.
- **The other bot runs out of market data.** Both share IB's per-account limit
  on simultaneous quotes. Lower `chain_strikes` / `chain_expirations`, or
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
