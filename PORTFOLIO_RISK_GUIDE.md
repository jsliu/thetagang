# Portfolio Risk Guide

How to read the risk tables the discretionary strategy (`run.strategies =
["discretionary"]`, see `thetagang.discretionary.toml`) prints at the start of each
run, next to the **Account summary**:

1. **Portfolio greeks (SPX beta-weighted)**: overall exposure
2. **Stress P&L**: what happens on big market moves
3. **Delta by symbol**: which positions cause it
4. **Discretionary trades**: the new trades the bot will open, if any are configured

All three cover every **USD stock and option** position in the account.
Futures, non-USD positions and other contract types are not included.
**VIX options are left out too:** VIX moves against the market and its options
price off VIX futures, so delta × VIX × beta isn't SPX exposure. A VIX hedge's
effect is therefore not in any of these tables, including the stress table;
check VIX positions separately in TWS. The bot only closes long VIX calls you
hold once VIX rises above `strategies.vix_call_hedge.close_hedges_when_vix_exceeds`.

---

## 1. Key ideas

### Three levels of delta

| Measure | Per position | Accounts for price? | Accounts for beta? |
|---|---|---|---|
| Raw delta, in shares | Δ | No | No |
| Delta dollars | Δ × price | Yes | No |
| Beta delta dollars | Δ × price × beta | Yes | Yes |

Here Δ is the share-equivalent delta: option delta × contracts × multiplier.
Stock has delta 1 per share. Short positions carry a negative sign.

**Raw delta can't be added across stocks.** For example, take long 10 deltas of
a $500 stock and short 10 deltas of a $10 stock:

| | $500 stock | $10 stock | Total |
|---|---|---|---|
| Raw delta | +10 | −10 | 0, looks neutral |
| Delta dollars | +$5,000 | −$100 | **+$4,900, actually long** |

Always judge exposure in delta dollars or beta delta dollars, never raw delta.

### Beta

Beta measures how much a stock tends to move when the market moves. A beta of
1.5 means the stock tends to move 1.5% when SPX moves 1%.

How it's calculated here:

```
beta = covariance(stock daily log returns, SPX daily log returns)
       ÷ variance(SPX daily log returns)
```

- **Lookback.** One year of daily closes from IB. IB's own default is two
  years, so IB's numbers will differ.
- **SPX and XSP.** Beta is 1.
- **Missing data.** If a symbol has fewer than 20 shared trading days with SPX,
  or its history fails to load, beta defaults to 1.

### Beta delta dollars = SPX delta × SPX price

```
beta delta $ = Σ  beta × Δ × price      (summed over all positions)
SPX delta    = beta delta $ ÷ SPX price
```

They are the same measurement in different units. One is in dollars, the other
in "units of SPX". When one is zero, so is the other.

**Meaning:** beta delta dollars is your S&P 500 exposure in dollars. A reading
of $42,842 means the portfolio tends to move like $42,842 invested in the S&P 500:

| S&P 500 move | Expected portfolio change |
|---|---|
| +1% | about +$428 |
| −1% | about −$428 |
| −5% | about −$2,142 |

This only holds for small moves. Large moves change option deltas, which is
gamma. See the stress test below.

### How these compare with IB's columns

| IB TWS column | IB's definition | In this table |
|---|---|---|
| Beta Delta Dollars | Exposure weighted by the unadjusted beta | **Beta delta $**, same formula |
| SPX (Beta Weighted) Delta | Delta × dollar-adjusted beta, using the ratio of close prices | Not reproduced; IB uses a different scaling |

Even with the same formula, the numbers won't match IB exactly:

- **Beta.** One year of log returns here, versus IB's two-year default or
  whatever you set in TWS. This is usually the biggest difference.
- **Positions included.** Only USD stocks and options are counted here.
  IB includes futures and others.
- **Snapshot timing.** Prices and deltas are taken when the run happens.

---

## 2. Portfolio greeks table

| Row | Meaning |
|---|---|
| SPX price | Current SPX index price |
| SPX delta | Beta delta $ ÷ SPX price: portfolio exposure in units of SPX |
| Beta delta $ | Σ beta × Δ × price |
| Delta $ (no beta) | Σ Δ × price: plain-dollar exposure, beta ignored |
| Gamma (SPX delta / pt) | How much SPX delta changes for a 1-point SPX move |
| Theta ($/day) | Daily time decay of all options. Positive means you earn it. |
| Vega ($/vol pt) | Profit or loss if implied volatility rises 1 point on all options |
| Options without greeks | Options IB returned no Greeks for. These are left out of the totals. |

Option Greeks come straight from IB's model Greeks. Only beta and the totals are
calculated here.

**Two kinds of neutral:**

| Target | Protects against | Relies on |
|---|---|---|
| SPX delta = 0, the same as beta delta $ = 0 | Broad market moves | Betas being right |
| Delta $ (no beta) = 0 | All stocks moving by the same percentage | Nothing, it ignores beta |

Aim for SPX delta near zero, and use delta $ as a check. If the two are far
apart, your hedge depends heavily on the beta estimates.

---

## 3. Stress P&L table

Estimated profit and loss if SPX instantly moves −10%, −5%, +5% or +10%.

### The two columns

- **Each stock moves beta ×**: each stock moves by its beta × the SPX move.
- **All betas = 1**: every stock moves by exactly the SPX move. If SPX rises
  1%, every stock rises 1%.

For example, with SPX down 10%:

| Stock | Beta | Each stock moves beta × | All betas = 1 |
|---|---|---|---|
| A | 2.0 | −20% | −10% |
| B | 0.5 | −5% | −10% |
| C | 1.0 | −10% | −10% |

Stocks gain or lose position value × move. Options are **fully repriced** with
Black-Scholes at the new stock price and the option's current implied volatility
from IB. That captures curvature, which a single delta number misses. If IB has
no implied volatility for an option, a delta-plus-gamma estimate is used instead.

### How to read it

1. **Uneven rows mean short gamma.** If the −10% loss is much bigger than the
   +10% gain, or more than twice the −5% loss, big moves hurt you even when
   delta is close to zero.
2. **A gap between the columns means beta risk.** The gap is the difference
   between the two columns in the same row. It shows how much your result
   depends on the betas being right. That is the correlation risk Taleb warns
   about in *Dynamic Hedging*: betas come from history and tend to break down
   in a crisis.

### Worked example

| SPX −10% | P&L |
|---|---|
| Each stock moves beta × | −$6,053 |
| All betas = 1 | +$397 |
| Gap | about $6,450 |

- **In plain dollars, the book is roughly balanced.** If every stock fell
  exactly 10%, you'd roughly break even. +$397 is close to zero, so it does
  not mean you are net short.
- **In market terms, the book is long.** If high-beta stocks fall harder than
  the market, you lose about $6,053.
- **So the longs sit in higher-beta names than the shorts.** The outcome of a
  selloff depends mostly on whether those betas hold.

To tell whether a small "All betas = 1" result means slightly short or long
gamma, compare it with the +10% row in the same column, and with the
"Delta $ (no beta)" row in the greeks table:

| +10%, all betas = 1 | Meaning |
|---|---|
| About the opposite of −10% | Slightly net short or long in plain dollars |
| Same sign as −10%, both positive | Long gamma: gains on big moves either way, e.g. long puts |
| Much more negative | Short gamma on the upside, e.g. short calls |

### Simplifications

- **Instant shock.** Implied volatility, interest rates and time are held
  fixed. In a real selloff implied volatility usually jumps, so short options
  would lose more than shown.
- **European pricing.** It uses a zero interest rate and no dividends. Deep
  in-the-money American puts may be slightly off.

---

## 4. Delta by symbol table

One row per underlying, combining stock and options, sorted with the largest
beta effect first. The last row shows totals.

| Column | Meaning |
|---|---|
| Beta | The one-year beta used for the symbol |
| Delta $ (no beta) | Σ Δ × price for that symbol |
| Beta delta $ | Σ beta × Δ × price for that symbol |
| Beta effect | Beta delta $ − delta $ |

**The beta effect explains the stress gap.** The total beta effect is what
separates the two stress columns. For a −10% SPX move, the gap is about
**10% of the total beta effect**. For example, a ~$6,450 gap means a total beta
effect of roughly $64,500. Options make this approximate, but it holds for the
main contributors.

How to act on the top rows:

| Beta effect | Usual cause | To shrink the gap |
|---|---|---|
| Large positive | Long exposure in a high-beta stock, or short exposure in a low-beta one | Trim the long, or add short delta in that symbol |
| Large negative | Short exposure in a high-beta stock, or long exposure in a low-beta one | Reduce the short, or trim the low-beta long |

---

## 5. Hedging to zero

- **Weights alone don't hedge.** Bringing SPX delta to zero means trading.
  "Total = 0" is one condition across many positions, so you need a rule, e.g.
  shrink every long equally, or change only a few positions.
- **An index hedge is the simplest.** It neutralizes delta without changing
  your other positions. To offset +7.39 SPX delta:

| Hedge | SPX delta per unit | Size |
|---|---|---|
| Short SPY shares | about −0.1 per share | about 74 shares |
| Short MES futures | −5 per contract | 1 contract leaves +2.39, 2 leave −2.61 |
| Long XSP puts | −10 × put delta | e.g. 2 puts near 0.37 delta |

- **Taleb's *Dynamic Hedging*.** The book's model is to hedge each underlying in
  that same underlying. Index or beta hedges create basis and correlation risk,
  and delta only describes small moves. Check every hedge with scenarios, like
  the stress table, not one number.
- **Neither target removes everything.** Stock-specific risk, such as earnings,
  remains. So do gamma, vega and theta.

---

## 6. Discretionary trades to SPX delta = 0

List the trades you want in `[[strategies.discretionary.trades]]` (see
`thetagang.discretionary.toml`): a symbol and a structure (`long_call`,
`short_call`, `long_put`, `short_put`, `put_credit_spread`, `call_credit_spread`,
`put_debit_spread`, `call_debit_spread`), the main leg's target delta and DTE, and
for verticals the `width` to the second leg. The bot decides **how many**.

**Existing positions never change size.** Only the listed trades are sized:

```
need           = −(SPX delta now)
SPX Δ per unit = Σ legs (sign × delta × multiplier) × underlying price × beta ÷ SPX price
contracts      = round(need × weight ÷ Σ weights ÷ SPX Δ per unit)
```

- **Weights** split the needed delta between trades. Only trades that move delta
  the right way share it: one that would push it the wrong way gets 0 (for
  example a long call when you're already long the market).
- **Whole units.** A vertical is sized in whole spreads, a single leg in whole
  contracts, so SPX delta lands near 0, not exactly on it.
- **`max_contracts`** caps a trade. A capped trade's leftover delta is not
  passed to the other trades, so SPX delta can land short of 0.
- **`delta_band`**: no new trades while |SPX delta| is at most this.
- **Missing greeks:** if IB returns no greeks for any option you hold, SPX delta
  is incomplete, so the plan is shown but no trades are opened that run.
- **Rolls and closes first:** trades are sized on the account before this run's
  rolls and closes. If any roll or close order is queued, new trades wait for the
  next run, so they're re-sized once those fill instead of over-hedging.
- Beta comes from the same 1-year history as the other tables (1.0 if missing).

The **Discretionary trades** table shows each trade's legs, SPX Δ per unit,
weight, contracts and a note when it gets none. The greeks table's **After
trades** column shows every greek as it would be once the trades fill.

**Orders.** Single legs are limit orders at mid; a vertical is one combo order at
the net mid (negative = credit). They go out in the `post_discretionary` stage
(with `--dry-run`, shown only). Each is tagged `thetagang-discretionary-entry`,
and any still working is cancelled at the next start, whatever
`runtime.account.cancel_orders` says, so a rerun re-plans instead of doubling up.
Spread rolls/closes and VIX hedge sells are tagged `thetagang-discretionary-manage`
and cancelled the same way.

**Limits:**

- **Delta drifts.** Sizing holds for small moves only; option deltas change as
  prices move. Check the stress table and the After trades column, not one number.
- **One-number hedge.** It only zeros SPX delta. Plain delta $, gamma, vega and
  stock-specific risk remain.
- **Fills.** A mid-priced order may not fill; nothing is chased.

---

## 7. Target weights for SPX delta = 0

These targets appear in two tables:

- **Managed symbols and target weights**: the "Target weight (SPX Δ=0)" column.
- **Positions summary**: "Weight now", "Target weight (SPX Δ=0)",
  "Target beta delta $", "Share-equiv change" and "Net target contracts", next
  to the share and option counts for each symbol.

**Net target contracts** names the exact trade. It follows the same spread
rules as rolling, for puts and calls alike:

| Position in the symbol | How it's adjusted | Example |
|---|---|---|
| Credit vertical: short and long, same type, expiry and quantity; long put below the short, or long call above | As whole spreads, keeping the width | `open 2 spread SBUX 20270219 255/265C`, or `close 1 spread HOOD 20261120 50/45P` |
| Calendar, diagonal or uneven quantities: a short with a same-type long that isn't a vertical | Left alone, except short calls fully covered by your shares: those count as standalone | `none (spread legs)` if nothing else in the symbol can be adjusted |
| Standalone option | One contract at a time: + buy, − sell | `+2 HOOD  261120P00050000` buys 2. If you're short it, that closes 2. |

In each symbol it picks the position with the most delta, since that moves delta
most per trade, and rounds to whole units:

```
units = share-equiv change ÷ (delta per contract or per spread)
```

**It never flips a position.** Closing is capped at what you hold: it won't sell
more of a long option than you own, buy back more of a short than you're short,
or close more spreads than you have. When the cap applies, the "After trades"
column shows the delta that remains (when no discretionary trades are configured).

For a symbol with stock but no adjustable option, it names a share trade such
as `-100 shares`, capped the same way: it never sells more stock than you hold
or buys back more than you're short. Rounding means you won't land exactly on zero. The
greeks table's **After trades** column shows every greek (SPX delta, beta
delta $, delta $, gamma, theta, vega) as it would be if you made every
suggested trade, next to the **Now** column. When
`[[strategies.discretionary.trades]]` are configured, that column shows the
planned discretionary trades instead (section 6), since those are real orders.

They are suggestions only and place no orders. A symbol shows "n/a" if its
Greeks couldn't be computed. Positions such as SGOV or VIX are counted in the
totals but have no row in these tables.

**Rule:** reach **both** zeros at once:

- SPX delta = 0, which is the same as Σ beta delta $ = 0: hedged against market moves
- Σ delta $ (no beta) = 0: hedged against all stocks moving the same percentage,
  so the hedge holds even if the betas are off

Many portfolios satisfy both, so there is no single answer. The table picks the
one with the **smallest proportional change**, using weighted least squares on
plain delta dollars:

```
minimize   Σ (change_i)² / |delta $_i|
subject to Σ delta $ = 0   and   Σ beta × delta $ = 0
```

Each symbol gets its own percentage change based on its size and beta. With
only the first condition, this is the same as changing every symbol by the same
percentage. Symbols with zero delta $ never change.

For example, two longs and one short:

| Symbol | Beta | Delta $ now | Target delta $ | Beta delta $ now | Target beta delta $ |
|---|---|---|---|---|---|
| HIGH | 2.0 | +$50,000 | +$23,305 | +$100,000 | +$46,609 |
| LOW | 0.5 | +$50,000 | +$46,609 | +$25,000 | +$23,305 |
| SHORT | 1.0 | −$57,158 | −$69,914 | −$57,158 | −$69,914 |
| Total | | +$42,842 | **$0** | +$67,842 | **$0** |

The high-beta long is cut most (−53%), the low-beta long only a little (−7%),
and the short grows (+22%). Both totals end at zero.

| Column | Meaning |
|---|---|
| Weight now / Target weight | \|Beta delta $\| ÷ Σ \|beta delta $\|, labeled long or short by market side. Weights are positive and sum to 100%. At target, long and short weights each sum to 50%. |
| Beta delta $ now / Target beta delta $ | Exposure before and after |
| Share-equiv change | Change in delta $ ÷ price: shares of stock to add (+) or remove (−). With options, divide by 100 × the option's delta to get contracts. |

**Limits:**

- **One-sided books.** If every symbol is on the same side, the rule flattens
  everything. An index hedge (section 5) is the better choice then.
- **Too few symbols.** With only two symbols, or symbols that all share one
  beta, both conditions may not be reachable without flattening. The greeks
  table's "After trades" column shows how close the suggestion gets.
- **Sign flips.** A symbol can be pushed from long to short if that's the
  smallest way to reach both zeros. Check any row whose side changes.
- **Smallest change only.** It doesn't consider which positions are cheapest to
  trade, or which you want to keep. Use it as a starting point.
- **Delta drifts.** Hitting the target in share-equivalents works for small
  moves only, because option deltas change as prices move.
