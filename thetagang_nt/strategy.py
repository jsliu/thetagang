"""Discretionary: thetagang's discretionary strategy as a Nautilus Strategy.

Every `interval_secs` (market hours) it:
  - rolls/closes short options and credit verticals with the [roll_when] rules,
    closes debit verticals at the take-profit and rolls them near expiry
    (rolls and closes are one IB combo order: a Nautilus OptionSpread),
  - runs the VIX call hedge (buy per VIXMO allocation, sell above a VIX level),
  - opens the configured trades, sized to bring SPX beta-weighted delta to 0,
    when nothing was rolled this run and |SPX delta| is outside delta_band.
Unfilled orders are moved halfway to the mid once, after reprice_after_secs.
With dry_run (the default) orders are logged and counted, never sent."""

from __future__ import annotations

import functools
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from nautilus_trader.common.events import TimeEvent
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import OptionKind, OrderSide, TimeInForce
from nautilus_trader.model.events import OrderAccepted
from nautilus_trader.model.identifiers import (
    ClientOrderId,
    InstrumentId,
    generic_spread_id_to_list,
    is_generic_spread_id,
    new_generic_spread_id,
)
from nautilus_trader.model.instruments import Instrument, OptionContract
from nautilus_trader.model.objects import Price
from nautilus_trader.trading.strategy import Strategy
from prometheus_client import Counter, Gauge

from thetagang_nt import aio
from thetagang_nt.config import DiscretionaryConfig, TradeConfig
from thetagang_nt.monitor import BETAS_KEY
from thetagang_nt.portfolio import (
    INDEX_CONTRACTS,
    SPX,
    VENUE,
    VIX,
    VIX_ROOT,
    VIXMO,
    bid,
    build_book,
    held_legs,
    is_index,
    option_greeks,
    price,
    stock_contract,
    underlying_id,
)
from thetagang_nt.rules import (
    STRUCTURES,
    Candidate,
    Leg,
    Plan,
    close_if_unable_to_roll,
    credit_pnl,
    debit_pnl,
    debit_take_profit_value,
    is_debit_vertical,
    is_itm,
    pick_contract,
    roll_strike_limit,
    short_pnl,
    should_close,
    should_roll,
    size_plans,
    split_option_legs,
    valid_strike,
    vix_hedge_weight,
)

NEW_YORK = ZoneInfo("America/New_York")
ORDERS = Counter(
    "thetagang_orders", "Orders placed by the strategy", ["action", "dry_run"]
)
LAST_RUN = Gauge(
    "thetagang_strategy_last_run_timestamp_seconds", "Last completed strategy run"
)
# config the alert rules need (Prometheus can't read the TOML)
DRY_RUN = Gauge("thetagang_strategy_dry_run", "1 while orders are only logged")
DELTA_BAND = Gauge(
    "thetagang_strategy_delta_band", "No new trades while |SPX delta| <= this"
)
# the trade plan, for the dashboard (the old "Discretionary trades" table)
PLAN = Gauge(
    "thetagang_plan_units",
    "Contracts (spreads: spreads) the plan opens per configured trade",
    ["trade", "contracts", "note"],
)
PLAN_UNIT = Gauge(
    "thetagang_plan_unit_spx_delta", "SPX delta added per unit", ["trade"]
)
PLAN_STATUS = Gauge("thetagang_plan_status", "Why the plan is what it is", ["status"])
AFTER = Gauge(
    "thetagang_greeks_after_trades",
    "Portfolio greeks after the planned trades (SPX beta-weighted)",
    ["greek"],
)


def plan_status(status: str) -> None:
    PLAN_STATUS.clear()
    PLAN_STATUS.labels(status=status).set(1)


ComboLegs = list[tuple[InstrumentId, int]]  # (leg, +1 buy / -1 sell per unit)


class NoContract(Exception):
    """Nothing suitable to trade: logged, the position is left for next run."""


@functools.cache
def _nyse() -> Any:  # building the calendar takes a second: once
    import exchange_calendars

    return exchange_calendars.get_calendar("XNYS")


def market_open(now: datetime) -> bool:
    return bool(_nyse().is_open_on_minute(pd.Timestamp(now).floor("min")))


def legs_of(instrument_id: InstrumentId) -> list[tuple[InstrumentId, int]]:
    if is_generic_spread_id(instrument_id):
        return list(generic_spread_id_to_list(instrument_id))
    return [(instrument_id, 1)]


class Discretionary(Strategy):
    def __init__(self, config: DiscretionaryConfig) -> None:
        super().__init__(config)
        self.rw = config.roll_when.rules()
        self._busy = False
        self._repriced: set[ClientOrderId] = set()

    # ------------------------------------------------------------ lifecycle

    def on_start(self) -> None:
        DRY_RUN.set(self.config.dry_run)
        DELTA_BAND.set(self.config.delta_band)
        if self.config.dry_run:
            self.log.warning("dry_run: orders are logged, never sent")
        aio.spawn(self, self._boot(), "strategy boot")

    def on_stop(self) -> None:
        self.clock.cancel_timers()

    async def _boot(self) -> None:
        await aio.load_contracts(self, INDEX_CONTRACTS)
        self.clock.set_timer(
            "discretionary",
            timedelta(seconds=self.config.interval_secs),
            callback=self._on_timer,
        )
        await self._run()

    def _on_timer(self, _event: TimeEvent) -> None:
        if not self._busy:
            aio.spawn(self, self._run(), "strategy run")

    async def _run(self) -> None:
        if self.config.market_hours_only and not market_open(self.clock.utc_now()):
            self.log.info("Market closed, nothing to do")
            return
        self._busy = True
        try:
            await self._cycle()
            LAST_RUN.set(self.clock.timestamp())
        finally:
            self._busy = False

    # ---------------------------------------------------------------- run

    def _today(self) -> date:
        return self.clock.utc_now().astimezone(NEW_YORK).date()

    async def _cycle(self) -> None:
        today = self._today()
        symbols = {leg.symbol for leg in held_legs(self.cache)}
        ids = {leg.instrument_id for leg in held_legs(self.cache)}
        ids |= {underlying_id(s) for s in symbols} | {SPX, VIX, VIXMO}
        await aio.wait_for_prices(self, ids, self.config.quote_timeout_secs)
        legs = held_legs(self.cache)
        busy = self._symbols_with_open_orders()

        by_symbol: dict[str, list[Leg]] = defaultdict(list)
        for leg in legs:
            by_symbol[leg.symbol].append(leg)
        acted = False
        for symbol, symbol_legs in sorted(by_symbol.items()):
            if is_index(symbol) or self._no_trading(symbol):
                continue  # index options: roll by hand; VIX: the hedge below
            if symbol in busy:
                self.log.info(f"{symbol}: open orders, skipped this run")
                continue
            try:
                acted |= await self._manage_symbol(symbol, symbol_legs, today)
            except Exception as e:  # noqa: BLE001 - one symbol never blocks the rest
                self.log.error(f"{symbol}: manage failed: {e!r}")
        try:
            acted |= await self._vix_hedge(legs, today, busy)
        except Exception as e:  # noqa: BLE001
            self.log.error(f"VIX hedge failed: {e!r}")
        if not self.config.trades:
            return
        if acted:
            self.log.warning("Rolls/closes this run change SPX delta: new trades wait")
            plan_status("rolls/closes this run: new trades wait")
            return
        await self._open_trades(legs, today, busy)

    def _symbols_with_open_orders(self) -> set[str]:
        out = set()
        for order in self.cache.orders_open():
            for iid, _ in legs_of(order.instrument_id):
                ins = self.cache.instrument(iid)
                is_option = isinstance(ins, OptionContract)
                out.add(ins.underlying if is_option else iid.symbol.value)
        return out

    def _no_trading(self, symbol: str) -> bool:
        s = self.config.symbols.get(symbol)
        return bool(s and s.no_trading)

    def _strike_limit(self, symbol: str, right: str) -> float | None:
        s = self.config.symbols.get(symbol)
        if s is None:
            return None
        return s.put_strike_limit if right == "P" else s.call_strike_limit

    def _roll_target(self, symbol: str) -> tuple[float, int, int | None]:
        t, s = self.config.roll_target, self.config.symbols.get(symbol)
        delta = s.delta if s and s.delta else t.delta
        max_dte = s.max_dte if s and s.max_dte else t.max_dte
        return delta, t.dte, max_dte

    # ----------------------------------------------------- held positions

    async def _manage_symbol(self, symbol: str, legs: list[Leg], today: date) -> bool:
        und = price(self.cache, underlying_id(symbol))
        if und is None:
            self.log.warning(f"{symbol}: no underlying price, skipped")
            return False
        stock = [leg for leg in legs if not leg.is_option and leg.qty > 0]
        share_cost = stock[0].avg_px if stock else None
        pairs, unpaired, mixed = split_option_legs(legs)
        acted = False
        for short, long in pairs:
            if is_debit_vertical(short, long):
                acted |= await self._debit_spread(symbol, short, long, today)
            else:
                acted |= await self._credit_spread(symbol, short, long, und, today)
        for leg in unpaired:
            if leg.qty >= 0:
                continue
            if leg.right in mixed:
                self.log.warning(f"{leg.instrument_id}: calendar/diagonal, by hand")
                continue
            acted |= await self._short_option(symbol, leg, und, today, share_cost)
        return acted

    async def _short_option(
        self, symbol: str, leg: Leg, und: float, today: date, share_cost: float | None
    ) -> bool:
        pnl = short_pnl(leg)
        if pnl is None:
            self.log.warning(f"{leg.instrument_id}: no quote, skipped")
            return False
        dte, itm = leg.dte(today), is_itm(leg.right, leg.strike, und)
        if should_roll(self.rw, leg.right, itm, dte, pnl):
            try:
                await self._roll_short(symbol, leg, itm, share_cost)
                return True
            except NoContract as e:
                self.log.warning(f"{leg.instrument_id}: roll failed: {e}")
                if close_if_unable_to_roll(self.rw, dte, pnl):
                    await self._close(symbol, [leg], "close (unable to roll)")
                    return True
                return False
        if should_close(self.rw, pnl):
            await self._close(symbol, [leg], "close")
            return True
        return False

    async def _roll_short(
        self, symbol: str, leg: Leg, itm: bool, share_cost: float | None
    ) -> None:
        configured = self._strike_limit(symbol, leg.right)
        if leg.right == "C" and share_cost:  # covered call: never below cost
            configured = max(configured or 0.0, share_cost)
        limit = roll_strike_limit(
            leg.right, leg.strike, leg.avg_px - leg.mid, itm, configured
        )
        mc = self.config.minimum_credit
        floor = leg.mid + mc if self.rw.credit_only[leg.right] else mc
        delta, dte, max_dte = self._roll_target(symbol)
        new = await self._find(
            symbol,
            leg.right,
            delta,
            dte,
            max_dte,
            floor,
            strike_limit=limit,
            not_before=leg.expiry,
            exclude={(leg.strike, leg.expiry)},
        )
        await self._submit(
            symbol,
            "roll",
            [(leg.instrument_id, +1), (new.instrument_id, -1)],
            abs(leg.qty),
            leg.mid - new.price,
        )

    async def _credit_spread(
        self, symbol: str, short: Leg, long: Leg, und: float, today: date
    ) -> bool:
        pnl = credit_pnl(short, long)
        if pnl is None:
            self.log.warning(
                f"{short.instrument_id}/{long.instrument_id}: no net credit or no "
                "quote, manage by hand"
            )
            return False
        dte, itm = short.dte(today), is_itm(short.right, short.strike, und)
        if should_roll(self.rw, short.right, itm, dte, pnl):
            try:
                await self._roll_credit_spread(symbol, short, long, itm)
                return True
            except NoContract as e:
                self.log.warning(f"{short.instrument_id}: spread roll failed: {e}")
                if close_if_unable_to_roll(self.rw, dte, pnl):
                    await self._close(
                        symbol, [short, long], "close spread (unable to roll)"
                    )
                    return True
                return False
        if should_close(self.rw, pnl):
            await self._close(symbol, [short, long], "close spread")
            return True
        return False

    async def _roll_credit_spread(
        self, symbol: str, short: Leg, long: Leg, itm: bool
    ) -> None:
        """One 4-leg combo, same width. With credit_only the new short must pay
        for the buyback, the new long and the minimum credit; the long's cost is
        only known after a search, so search again with the bar raised by it."""
        old_net = short.mid - long.mid
        limit = roll_strike_limit(
            short.right,
            short.strike,
            (short.avg_px - long.avg_px) - old_net,
            itm,
            self._strike_limit(symbol, short.right),
        )
        width = long.strike - short.strike
        credit_only, mc = self.rw.credit_only[short.right], self.config.minimum_credit
        delta, dte, max_dte = self._roll_target(symbol)
        exclude = {(short.strike, short.expiry), (long.strike, long.expiry)}
        old_ids = {short.instrument_id, long.instrument_id}
        long_cost, found = 0.0, None
        for _ in range(3):
            floor = old_net + mc + long_cost if credit_only else mc
            new_short = await self._find(
                symbol,
                short.right,
                delta,
                dte,
                max_dte,
                floor,
                strike_limit=limit,
                not_before=short.expiry,
                exclude=exclude,
            )
            new_long = self._option_at(
                symbol, short.right, new_short.expiry, new_short.strike + width
            )
            long_price = await self._price_of(new_long)
            if new_long is None or new_long in old_ids or long_price is None:
                exclude.add((new_short.strike, new_short.expiry))
                continue
            new_net = new_short.price - long_price
            found = (new_short.instrument_id, new_long, new_net)
            if not credit_only or old_net - new_net <= -mc or long_price <= long_cost:
                break
            long_cost = long_price
        if found is None:
            raise NoContract("no new spread at the same width")
        new_short_id, new_long_id, new_net = found
        combo_price = old_net - new_net  # BUY combo: negative = net credit
        if credit_only and combo_price > -mc:
            raise NoContract(
                f"roll costs {combo_price:.2f}, credit_only needs {mc:.2f}"
            )
        await self._submit(
            symbol,
            "roll spread",
            [
                (short.instrument_id, +1),
                (long.instrument_id, -1),
                (new_short_id, -1),
                (new_long_id, +1),
            ],
            abs(short.qty),
            combo_price,
        )

    async def _debit_spread(
        self, symbol: str, short: Leg, long: Leg, today: date
    ) -> bool:
        if short.mid is None or long.mid is None:
            self.log.warning(f"{long.instrument_id}: debit spread without quotes")
            return False
        value, dte = long.mid - short.mid, long.dte(today)
        tp = debit_take_profit_value(short, long, self.config.debit_spread_take_profit)
        if value >= tp:
            await self._close(symbol, [short, long], "close debit spread (take-profit)")
            return True
        if dte > self.rw.dte:
            return False
        try:
            await self._roll_debit_spread(symbol, short, long, value)
            return True
        except NoContract as e:
            pnl = debit_pnl(short, long)
            if pnl is not None and close_if_unable_to_roll(self.rw, dte, pnl):
                await self._close(
                    symbol, [short, long], "close debit spread (unable to roll)"
                )
                return True
            self.log.warning(f"{long.instrument_id}: not rolled ({e}), roll by hand")
            return False

    async def _roll_debit_spread(
        self, symbol: str, short: Leg, long: Leg, value: float
    ) -> None:
        """Sell the old spread, buy one further out (same width) as one combo;
        the new long at the configured debit trade's delta/DTE for the symbol."""
        delta, dte, max_dte = self._roll_target(symbol)
        for t in self.config.trades:
            right, sign, second = STRUCTURES[t.structure]
            if t.symbol == symbol and second and sign > 0 and right == long.right:
                delta, dte = t.delta, t.dte
        offset = short.strike - long.strike
        exclude = {(short.strike, short.expiry), (long.strike, long.expiry)}
        old_ids = {short.instrument_id, long.instrument_id}
        for _ in range(3):
            new_long = await self._find(
                symbol,
                long.right,
                delta,
                dte,
                max_dte,
                0.01,
                not_before=long.expiry,
                exclude=exclude,
            )
            new_short = self._option_at(
                symbol, long.right, new_long.expiry, new_long.strike + offset
            )
            short_price = await self._price_of(new_short)
            if new_short and new_short not in old_ids and short_price is not None:
                break
            exclude.add((new_long.strike, new_long.expiry))
        else:
            raise NoContract("every candidate overlapped the old spread")
        combo_price = (new_long.price - short_price) - value  # positive = net debit
        max_debit = self.config.debit_spread_max_roll_debit * abs(offset)
        if combo_price > max_debit:
            raise NoContract(f"roll costs {combo_price:.2f} > {max_debit:.2f}")
        await self._submit(
            symbol,
            "roll debit spread",
            [
                (long.instrument_id, -1),
                (short.instrument_id, +1),
                (new_long.instrument_id, +1),
                (new_short, -1),
            ],
            abs(long.qty),
            combo_price,
        )

    async def _close(self, symbol: str, legs: list[Leg], action: str) -> None:
        if any(leg.mid is None for leg in legs):
            raise NoContract("no quote to close at")
        combo = [(leg.instrument_id, +1 if leg.qty < 0 else -1) for leg in legs]
        combo_price = sum(r * leg.mid for (_, r), leg in zip(combo, legs))
        await self._submit(symbol, action, combo, abs(legs[0].qty), combo_price)

    # ----------------------------------------------------------- VIX hedge

    async def _vix_hedge(self, legs: list[Leg], today: date, busy: set[str]) -> bool:
        cfg = self.config.vix_call_hedge
        if not cfg.enabled or VIX_ROOT in busy:
            return False
        vix, vixmo = price(self.cache, VIX), price(self.cache, VIXMO)
        calls = [c for c in legs if c.symbol == VIX_ROOT and c.right == "C"]
        held = sum(c.qty for c in calls if c.dte(today) > cfg.ignore_dte)
        threshold = cfg.close_hedges_when_vix_exceeds
        close = bool(threshold) and vix is not None and vix > threshold
        if held > 0:
            if not close:
                return False
            for leg in (c for c in calls if c.qty > 0):
                px = bid(self.cache, leg.instrument_id)
                if px is None:  # never sell a hedge at a no-quote price
                    self.log.error(f"{leg.instrument_id}: no bid, kept this run")
                    continue
                await self._submit(
                    "VIX",
                    f"sell VIX hedge (VIX {vix:.2f})",
                    [(leg.instrument_id, -1)],
                    leg.qty,
                    -px,
                )
            return True
        if close or vix is None:
            return False
        if vixmo is None:
            self.log.warning("VIX hedge: no VIXMO price")
            return False
        weight = vix_hedge_weight(cfg.allocation, vixmo)
        nlv = self._net_liquidation()
        if weight <= 0 or not nlv:
            return False
        c = await self._find(
            VIX_ROOT, "C", cfg.delta, cfg.target_dte, None, self.config.minimum_credit
        )
        px = bid(self.cache, c.instrument_id) or c.price
        qty = math.floor(weight * nlv / px / 100)
        if qty < 1:
            return False
        await self._submit(
            "VIX",
            f"buy VIX hedge (VIXMO {vixmo:.2f})",
            [(c.instrument_id, +1)],
            qty,
            px,
        )
        return True

    def _net_liquidation(self) -> float | None:
        for acc in self.cache.accounts():
            total = acc.balance_total(USD)
            if total is not None:
                return total.as_double()
        return None

    # ---------------------------------------------------------- new trades

    async def _open_trades(self, legs: list[Leg], today: date, busy: set[str]) -> None:
        cfg = self.config
        PLAN.clear()
        PLAN_UNIT.clear()
        AFTER.clear()
        spx = price(self.cache, SPX)
        if not spx:
            self.log.warning("No SPX price: no new trades")
            plan_status("no SPX price")
            return
        raw = self.cache.get(BETAS_KEY)
        betas = json.loads(raw) if raw else {}
        book = build_book(self.cache, self.greeks, legs, betas, spx, today)
        now = book.spx_delta
        if book.missing:
            self.log.warning(
                f"{len(book.missing)} position(s) without greeks: no trades"
            )
            plan_status(f"{len(book.missing)} position(s) without greeks: no trades")
            return
        if abs(now) <= cfg.delta_band:
            self.log.info(f"SPX delta {now:+.2f} within the band of {cfg.delta_band:g}")
            plan_status(f"SPX delta {now:+.2f} within the band of {cfg.delta_band:g}")
            return
        plans: list[tuple[TradeConfig, Plan, ComboLegs, float, dict[str, float]]] = []
        for t in cfg.trades:
            plan, combo, combo_price, unit = (
                Plan(t.weight, t.max_contracts),
                [],
                0.0,
                {},
            )
            if t.symbol in busy or self._no_trading(t.symbol):
                plan.note = "open orders or no_trading"
            else:
                try:
                    combo, combo_price, unit = await self._price_trade(
                        t, spx, betas.get(t.symbol, 1.0), today
                    )
                    plan.unit_spx = unit["beta_delta_dollars"] / spx
                    plan.priced = True
                except (NoContract, TimeoutError) as e:
                    plan.note = f"skipped: {e}"
            plans.append((t, plan, combo, combo_price, unit))
        size_plans([p for _, p, _, _, _ in plans], need=-now)
        self.log.info(f"SPX delta {now:+.2f}: sizing {len(plans)} trade(s) toward 0")
        after = {
            "beta_delta_dollars": book.beta_dollars,
            "delta_dollars": book.plain_dollars,
            "gamma": book.gamma,
            "theta": book.theta,
            "vega": book.vega,
            "volga": book.volga,
        }
        for t, plan, combo, _, unit in plans:
            name = f"{t.symbol} {t.structure}"
            contracts = " / ".join(
                f"{'+' if r > 0 else '-'}{i.symbol.value}" for i, r in combo
            )
            PLAN.labels(trade=name, contracts=contracts or "-", note=plan.note).set(
                plan.units
            )
            if plan.priced:
                PLAN_UNIT.labels(trade=name).set(plan.unit_spx)
            for greek, value in unit.items():
                after[greek] += plan.units * value
        AFTER.labels(greek="spx_delta").set(after["beta_delta_dollars"] / spx)
        for greek, value in after.items():
            AFTER.labels(greek=greek).set(value)
        plan_status(f"SPX delta {now:+.2f} -> {after['beta_delta_dollars'] / spx:+.2f}")
        for t, plan, combo, combo_price, _ in plans:
            self.log.info(
                f"{t.symbol} {t.structure}: {plan.units} unit(s) at "
                f"{plan.unit_spx:+.3f} SPX delta each. {plan.note}"
            )
            if plan.units:
                await self._submit(
                    t.symbol, f"open {t.structure}", combo, plan.units, combo_price
                )

    async def _price_trade(
        self, t: TradeConfig, spx: float, beta: float, today: date
    ) -> tuple[ComboLegs, float, dict[str, float]]:
        """(combo legs, price per unit, greeks added per unit) of one configured
        trade: delta $, beta delta $, SPX gamma, theta, vega."""
        right, sign, second = STRUCTURES[t.structure]
        main = await self._find(t.symbol, right, t.delta, t.dte, None, 0.01)
        picks = [(main.instrument_id, sign, main.price)]
        if second is not None:
            direction, sign2 = second
            strike = main.strike + direction * t.width
            iid = self._option_at(t.symbol, right, main.expiry, strike)
            other = await self._price_of(iid)
            if iid is None or other is None:
                raise NoContract(f"no quote for the second leg at {strike:g}")
            picks.append((iid, sign2, other))
        und = price(self.cache, underlying_id(t.symbol))
        unit = dict.fromkeys(
            ("beta_delta_dollars", "delta_dollars", "gamma", "theta", "vega", "volga"),
            0.0,
        )
        for iid, s, _ in picks:
            g = option_greeks(
                self.cache, self.greeks, self.cache.instrument(iid), today
            )
            if g is None:
                raise NoContract(f"no greeks for {iid.symbol.value}")
            size = s * 100
            unit["delta_dollars"] += g.delta * size * und
            unit["beta_delta_dollars"] += g.delta * size * und * beta
            unit["gamma"] += g.gamma * size * (beta * und / spx) ** 2
            unit["theta"] += g.theta * size
            unit["vega"] += g.vega * size
            unit["volga"] += g.volga * size
        combo_price = sum(s * p for _, s, p in picks)
        return [(iid, s) for iid, s, _ in picks], combo_price, unit

    # ------------------------------------------------------ chain scanning

    def _options(self, symbol: str, right: str) -> list[OptionContract]:
        kind = OptionKind.CALL if right == "C" else OptionKind.PUT
        return [
            i
            for i in self.cache.instruments(VENUE, underlying=symbol)
            if isinstance(i, OptionContract) and i.option_kind == kind
        ]

    def _option_at(
        self, symbol: str, right: str, expiry: date, strike: float
    ) -> InstrumentId | None:
        for ins in self._options(symbol, right):
            if ins.expiration_utc.date() == expiry and math.isclose(
                float(ins.strike_price), strike, abs_tol=1e-6
            ):
                return ins.id
        return None

    async def _price_of(self, iid: InstrumentId | None) -> float | None:
        if iid is None:
            return None
        timeout = self.config.quote_timeout_secs
        return (await aio.wait_for_prices(self, [iid], timeout))[iid]

    async def _find(
        self,
        symbol: str,
        right: str,
        delta: float,
        dte: int,
        max_dte: int | None,
        min_price: float,
        *,
        strike_limit: float | None = None,
        not_before: date | None = None,
        exclude: set[tuple[float, date | None]] | None = None,
    ) -> Candidate:
        """The old scanner's pick: load the chain from IB, quote the strikes
        nearest the money in the first expirations, compute Nautilus greeks."""
        cfg, exclude, today = self.config, exclude or set(), self._today()
        min_days = max(dte, (not_before - today).days if not_before else 0)
        max_days = max_dte if max_dte and max_dte >= min_days else min_days + 60
        base = (
            {"secType": "IND", "symbol": symbol[1:], "exchange": "CBOE"}
            if is_index(symbol)
            else stock_contract(symbol)
        )
        chain_range = {
            "build_options_chain": True,
            "min_expiry_days": min_days,
            "max_expiry_days": max_days,
        }
        await aio.load_contracts(self, [base | chain_range])
        und = await self._price_of(underlying_id(symbol))
        if und is None:
            raise NoContract(f"no {symbol} price")
        chain = [
            i
            for i in self._options(symbol, right)
            if min_days <= (i.expiration_utc.date() - today).days <= max_days
            and (float(i.strike_price), i.expiration_utc.date()) not in exclude
            and valid_strike(right, float(i.strike_price), strike_limit, und)
        ]
        expiries = sorted({i.expiration_utc.date() for i in chain})
        expiries = expiries[: cfg.chain_expirations]
        scan: list[OptionContract] = []
        for expiry in expiries:
            nearest_first = sorted(
                (i for i in chain if i.expiration_utc.date() == expiry),
                key=lambda i: float(i.strike_price),
                reverse=right == "P",
            )
            scan += nearest_first[: cfg.chain_strikes]
        if not scan:
            raise NoContract(f"{symbol}: no {right} strikes/expirations in range")
        forwards = await self._vix_forwards(expiries, und) if symbol == VIX_ROOT else {}
        await aio.wait_for_prices(self, [i.id for i in scan], cfg.quote_timeout_secs)
        candidates = []
        for ins in scan:
            expiry = ins.expiration_utc.date()
            g = option_greeks(self.cache, self.greeks, ins, today, forwards.get(expiry))
            candidates.append(
                Candidate(
                    ins.id,
                    float(ins.strike_price),
                    expiry,
                    (expiry - today).days,
                    g.delta if g else None,
                    price(self.cache, ins.id),
                )
            )
        pick = pick_contract(candidates, right, und, delta, min_price)
        keep = {leg.instrument_id for leg in held_legs(self.cache)}
        if pick is not None:
            keep.add(pick.instrument_id)
        for ins in scan:
            if ins.id not in keep:
                aio.unsubscribe(self, ins.id)
        if pick is None:
            raise NoContract(
                f"{symbol}: no {right} with |delta| <= {delta} priced above {min_price:g}"
            )
        self.log.info(
            f"{symbol}: picked {pick.instrument_id} delta={pick.delta:+.2f} "
            f"price={pick.price:.2f} dte={pick.dte}"
        )
        return pick

    async def _vix_forwards(
        self, expiries: list[date], spot: float
    ) -> dict[date, float]:
        """VIX future per expiry from put-call parity at the strike nearest spot."""
        out = {}
        for expiry in expiries:
            pair = []
            for right in ("C", "P"):
                same = [
                    i
                    for i in self._options(VIX_ROOT, right)
                    if i.expiration_utc.date() == expiry
                ]
                if same:
                    pair.append(
                        min(same, key=lambda i: abs(float(i.strike_price) - spot))
                    )
            if len(pair) != 2 or pair[0].strike_price != pair[1].strike_price:
                continue
            timeout = self.config.quote_timeout_secs
            prices = await aio.wait_for_prices(self, [i.id for i in pair], timeout)
            call, put = prices[pair[0].id], prices[pair[1].id]
            if call is not None and put is not None:
                out[expiry] = call - put + float(pair[0].strike_price)
        return out

    # -------------------------------------------------------------- orders

    @staticmethod
    def _tick_price(ins: Instrument, value: float) -> Price:
        inc = ins.price_increment.as_double()
        return ins.make_price(round(value / inc) * inc)

    async def _submit(
        self, symbol: str, action: str, combo: ComboLegs, qty: float, combo_price: float
    ) -> None:
        """One limit order at mid: a plain order for one leg, else a BUY of the
        Nautilus OptionSpread (IB combo) at the net mid (negative = credit)."""
        qty = int(abs(qty))
        if qty < 1:
            return
        desc = " / ".join(f"{'+' if r > 0 else '-'}{i.symbol.value}" for i, r in combo)
        dry = self.config.dry_run
        self.log.warning(
            f"{'[DRY RUN] ' if dry else ''}{symbol}: {action}: "
            f"{qty} x {desc} @ {combo_price:+.2f}"
        )
        ORDERS.labels(action=action.split(" (")[0], dry_run=str(dry).lower()).inc()
        if dry:
            return
        if len(combo) == 1:
            iid, ratio = combo[0]
            side = OrderSide.BUY if ratio > 0 else OrderSide.SELL
            limit = abs(combo_price)
        else:
            iid, side, limit = new_generic_spread_id(combo), OrderSide.BUY, combo_price
            await aio.load_instrument(self, iid)
        ins = self.cache.instrument(iid)
        if ins is None:
            raise NoContract(f"IB has no instrument for {iid}")
        order = self.order_factory.limit(
            iid,
            side,
            ins.make_qty(qty),
            self._tick_price(ins, limit),
            time_in_force=TimeInForce.DAY,
        )
        self.submit_order(order)

    def on_order_accepted(self, event: OrderAccepted) -> None:
        delay = self.config.reprice_after_secs
        if delay:
            self.clock.set_time_alert(
                f"reprice-{event.client_order_id}",
                self.clock.utc_now() + timedelta(seconds=delay),
                self._reprice,
            )

    def _reprice(self, event: TimeEvent) -> None:
        """Move an unfilled order halfway to the current mid, once."""
        coid = ClientOrderId(event.name.removeprefix("reprice-"))
        order = self.cache.order(coid)
        if order is None or not order.is_open or coid in self._repriced:
            return
        legs = legs_of(order.instrument_id)
        mids = [price(self.cache, iid) for iid, _ in legs]
        if any(m is None for m in mids):
            return
        mid = sum(r * m for (_, r), m in zip(legs, mids))
        ins = self.cache.instrument(order.instrument_id)
        new = self._tick_price(ins, (order.price.as_double() + mid) / 2)
        if new != order.price:
            self._repriced.add(coid)
            self.log.info(f"{order.instrument_id}: re-pricing {order.price} -> {new}")
            self.modify_order(order, price=new)
