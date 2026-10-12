"""Orders the discretionary strategy opens to bring SPX delta to 0.

DiscretionaryTradesMixin is mixed into DiscretionaryEngine. It sizes the trades in
strategies.discretionary.trades (new positions), and with place_target_orders turns
the positions summary's "Net target contracts" into orders (resizing what is
held). Orders are kept in planned_orders and enqueued by the post_discretionary
stage.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ib_async import PortfolioItem, Ticker, util
from ib_async.contract import Contract, Option, Stock
from ib_async.order import LimitOrder
from rich.table import Table

from thetagang import log
from thetagang.config import (
    DISCRETIONARY_STRUCTURES,
    Config,
    DiscretionaryTradeConfig,
)
from thetagang.ibkr import IBKR, TickerField
from thetagang.risk import (
    Adjuster,
    Book,
    Totals,
    is_debit_vertical,
    valid_greeks,
)
from thetagang.trading_operations import (
    NoValidContractsError,
    OptionChainScanner,
    OrderOperations,
)
from thetagang.util import (
    midpoint_or_market_price,
)


def quoted_price(ticker: Ticker, name: str) -> float:
    """midpoint_or_market_price, but raise instead of returning the 0.0 fallback."""
    price = midpoint_or_market_price(ticker)
    if not price or util.isNan(price) or price <= 0:
        raise RuntimeError(f"no quote for {name}")
    return price


# orderRef of orders that open discretionary trades: stale ones are always
# cancelled at startup so a rerun can't open the same trade twice
ENTRY_ORDER_REF = "thetagang-discretionary-entry"
# orderRef of spread rolls/closes and VIX hedge sells: also always cancelled at
# startup, so a rerun with cancel_orders = false can't send them twice
MANAGE_ORDER_REF = "thetagang-discretionary-manage"
# orderRef of the "Net target contracts" orders (place_target_orders)
RESIZE_ORDER_REF = "thetagang-discretionary-resize"
DISCRETIONARY_ORDER_REFS = (ENTRY_ORDER_REF, MANAGE_ORDER_REF, RESIZE_ORDER_REF)


@dataclass
class TradeLeg:
    contract: Contract
    sign: int  # +1 long, -1 short, per unit of the trade
    greeks: Any
    price: float  # mid


@dataclass
class Resize:
    """One "Net target contracts" trade: units of `adjuster` (+ buy / open more,
    - sell / close), or shares of `stock` when there is no adjustable option."""

    symbol: str
    units: int
    adjuster: Adjuster | None = None
    stock: PortfolioItem | None = None


@dataclass
class PlannedTrade:
    trade: DiscretionaryTradeConfig
    legs: list[TradeLeg] = field(default_factory=list)
    unit_share_delta: float = 0.0
    unit_plain_dollars: float = 0.0
    unit_beta_dollars: float = 0.0
    unit_spx: float = 0.0
    unit_gamma: float = 0.0
    unit_theta: float = 0.0
    unit_vega: float = 0.0
    units: int = 0
    note: str = ""


def size_plans(plans: list[PlannedTrade], need: float) -> None:
    """Split `need` (SPX delta) across the priced trades by weight, in whole units,
    never in the wrong direction, capped by max_contracts. A capped trade's
    remainder is not passed to the others."""
    usable = [p for p in plans if p.legs and p.unit_spx and p.unit_spx * need > 0]
    for p in plans:
        if p.legs and p.unit_spx and p not in usable:
            p.note = p.note or "would move SPX delta the wrong way"
    total_weight = sum(p.trade.weight for p in usable)
    for p in usable:
        target = need * p.trade.weight / total_weight
        p.units = max(round(target / p.unit_spx), 0)
        cap = p.trade.max_contracts
        if cap is not None and p.units > cap:
            p.units = cap
            p.note = p.note or f"capped at max_contracts={cap}"
        if p.units == 0:
            p.note = p.note or "less than one contract needed"


def adds_short_options(resize: Resize) -> bool:
    """A resize that sells more of a short option or opens more credit spreads:
    these need margin the bot doesn't check, so they're never placed."""
    adjuster = resize.adjuster
    if adjuster is None or resize.units == 0:
        return False
    if adjuster.is_spread:
        return resize.units > 0 and not is_debit_vertical(*adjuster.legs)
    return resize.units < 0 and adjuster.legs[0].position < 0


class DiscretionaryTradesMixin:
    """Needs the DiscretionaryEngine / DiscretionaryRiskMixin members declared below."""

    config: Config
    ibkr: IBKR
    order_ops: OrderOperations
    option_scanner: OptionChainScanner
    qualified_contracts: dict[int, Contract]
    planned_orders: list[tuple[Contract, LimitOrder]]
    resizes: list[Resize]
    resize_note: str

    if TYPE_CHECKING:

        def symbol_config(self, symbol: str) -> Any: ...

        async def beta(
            self, symbol: str, limit: asyncio.Semaphore | None = None
        ) -> float | None: ...

        async def leg_price(self, contract: Contract, exchange: str) -> float: ...

        def combo_order(
            self,
            symbol: str,
            legs: list[tuple[Contract, str]],
            qty: int,
            price: float,
            exchange: str,
        ) -> tuple[Contract, LimitOrder]: ...

    # ------------------------------------------------------- resize orders

    async def _resize_orders(self, book: Book, spx_price: float) -> None:
        """place_target_orders: turn the "Net target contracts" into orders, with
        the same guards as new trades (greeks complete, outside delta_band).
        resize_note says what happened, for the positions summary caption."""
        band = self.config.strategies.discretionary.delta_band
        now = book.totals.delta_dollars / spx_price
        if book.missing:
            self.resize_note = (
                f"place_target_orders: no orders this run ({book.missing} option(s) "
                "without greeks)."
            )
            log.warning(self.resize_note)
            return
        if abs(now) <= band:
            self.resize_note = (
                f"place_target_orders: no orders this run (SPX delta {now:+.2f} is "
                f"within the band of {band:g})."
            )
            log.info(self.resize_note)
            return
        exchange = self.order_ops.get_order_exchange()
        planned = skipped = 0
        for resize in self.resizes:
            if resize.symbol not in self.config.portfolio.symbols:
                log.warning(
                    f"{resize.symbol}: not managed by the bot (index option?), "
                    "target order not placed"
                )
                skipped += 1
                continue
            if adds_short_options(resize):
                log.warning(
                    f"{resize.symbol}: target order not placed - it would add short "
                    "options (naked or credit spread) without a margin check; place "
                    "it by hand if you want it"
                )
                skipped += 1
                continue
            try:
                self.planned_orders.append(await self._resize_order(resize, exchange))
                planned += 1
            except Exception as e:  # noqa: BLE001
                log.error(f"{resize.symbol}: target order skipped: {e}")
                skipped += 1
        self.resize_note = (
            f"place_target_orders: {planned} order(s) planned, {skipped} not placed "
            "(see the log). Planned orders go out in the post_discretionary stage, "
            "unless rolls/closes are queued that run (then they wait a run)."
        )

    async def _resize_order(
        self, resize: Resize, exchange: str
    ) -> tuple[Contract, LimitOrder]:
        """Limit order at mid: shares, one option, or a whole-spread combo."""
        if resize.adjuster is None or not resize.adjuster.is_spread:
            item = resize.stock if resize.adjuster is None else resize.adjuster.legs[0]
            assert item is not None
            contract = copy.copy(item.contract)
            contract.exchange = exchange
            actions = [(contract, "BUY" if resize.units > 0 else "SELL")]
        else:
            short, long = resize.adjuster.legs
            opening = resize.units > 0  # open more spreads: sell short, buy long
            actions = [
                (short.contract, "SELL" if opening else "BUY"),
                (long.contract, "BUY" if opening else "SELL"),
            ]
        legs = [(c, a, await self.leg_price(c, exchange)) for c, a in actions]
        return self._priced_order(
            resize.symbol, legs, abs(resize.units), RESIZE_ORDER_REF, exchange
        )

    def _priced_order(
        self,
        symbol: str,
        legs: list[tuple[Contract, str, float]],
        qty: int,
        order_ref: str,
        exchange: str,
    ) -> tuple[Contract, LimitOrder]:
        """(contract, BUY/SELL, mid) legs -> one limit order at mid: a plain order
        for one leg, else one BUY combo at the net mid (negative = credit)."""
        if len(legs) == 1:
            contract, action, mid = legs[0]
            order = self.order_ops.create_limit_order(
                action=action,
                quantity=qty,
                limit_price=round(mid, 2),
                order_ref=order_ref,
                transmit=True,
            )
            self.qualified_contracts[contract.conId] = contract
            return contract, order
        price = sum(mid if action == "BUY" else -mid for _, action, mid in legs)
        combo, order = self.combo_order(
            symbol, [(c, a) for c, a, _ in legs], qty, price, exchange
        )
        order.orderRef = order_ref
        return combo, order

    # ------------------------------------------------------- new trades

    async def _trade_legs(self, trade: DiscretionaryTradeConfig) -> list[TradeLeg]:
        """The contracts for one configured trade: the main leg at the target delta
        and, for a vertical, the second leg `width` away in the same expiry."""
        right, main_sign, second = DISCRETIONARY_STRUCTURES[trade.structure]
        sconfig = self.symbol_config(trade.symbol)
        underlying = Stock(
            trade.symbol,
            self.order_ops.get_order_exchange(),
            "USD",
            primaryExchange=sconfig.primary_exchange if sconfig else "",
        )
        ticker = await self.option_scanner.find_eligible_contracts(
            underlying,
            right,
            None,
            minimum_price=lambda: 0.01,
            target_dte=trade.dte,
            target_delta=trade.delta,
            # a new position: open interest must be known, as for the wheel
            allow_missing_open_interest=False,
            # max_dte is a roll rule: a new trade takes the nearest expiry at
            # least `dte` days out, with no cap (0 = no cap)
            max_dte=0,
        )
        main = ticker.contract
        if not main:
            raise NoValidContractsError("no contract at the target delta")
        legs = [
            TradeLeg(
                main,
                main_sign,
                ticker.modelGreeks,
                quoted_price(ticker, main.localSymbol),
            )
        ]
        if second is not None:
            offset, sign = second
            legs.append(await self._second_leg(main, offset * (trade.width or 0), sign))
        if not all(valid_greeks(leg.greeks) for leg in legs):
            raise NoValidContractsError("no option greeks from IB")
        return legs

    async def _second_leg(self, main: Contract, offset: float, sign: int) -> TradeLeg:
        """The vertical's other leg: same expiry and class, strike moved by offset."""
        other = await self.qualify_strike(
            main, round(main.strike + offset, 2), self.order_ops.get_order_exchange()
        )
        ticker = await self.ibkr.get_ticker_for_contract(
            other,
            required_fields=[],
            optional_fields=[
                TickerField.GREEKS,
                TickerField.MIDPOINT,
                TickerField.MARKET_PRICE,
            ],
        )
        return TradeLeg(
            other, sign, ticker.modelGreeks, quoted_price(ticker, other.localSymbol)
        )

    async def qualify_strike(
        self, like: Contract, strike: float, exchange: str
    ) -> Contract:
        """The option like `like` (same symbol, right, expiry, class) at `strike`."""
        qualified = await self.ibkr.qualify_contracts(
            Option(
                like.symbol,
                like.lastTradeDateOrContractMonth,
                strike,
                like.right,
                exchange,
                multiplier=like.multiplier,
                currency="USD",
                tradingClass=like.tradingClass,
            )
        )
        found = qualified[0] if qualified else None
        if not found or not found.conId:
            raise NoValidContractsError(
                f"no {like.right} at strike {strike} for "
                f"{like.lastTradeDateOrContractMonth}"
            )
        return found

    async def plan_trades(
        self, book: Book, betas: dict[str, float], spx_price: float
    ) -> tuple[Table | None, Totals]:
        """Size strategies.discretionary.trades so the book's SPX delta goes to 0.
        Existing positions are taken as they are and never resized. Orders are kept
        in self.planned_orders for the post_discretionary stage."""
        after = Totals(**vars(book.totals))
        cfg = self.config.strategies.discretionary
        if not cfg.trades:
            return None, after
        now = book.totals.delta_dollars / spx_price
        if book.missing:
            # SPX delta is incomplete without them: sizing against it could
            # open the wrong amount, so trade nothing (and skip the chain scans)
            blocked = f"{book.missing} option(s) without greeks: no new trades"
        elif abs(now) <= cfg.delta_band:
            blocked = f"SPX delta {now:+.2f} is within the band of {cfg.delta_band:g}"
        else:
            blocked = ""
        if blocked:
            plans = [PlannedTrade(t, note=blocked) for t in cfg.trades]
        else:
            # ponytail: one trade at a time to stay within IB pacing limits
            plans = [await self._price_trade(t, spx_price) for t in cfg.trades]
            size_plans(plans, need=-now)

        for p in plans:
            if not p.units:
                continue
            after.delta_dollars += p.units * p.unit_beta_dollars
            after.plain_delta_dollars += p.units * p.unit_plain_dollars
            after.gamma += p.units * p.unit_gamma
            after.theta += p.units * p.unit_theta
            after.vega += p.units * p.unit_vega
            self.planned_orders.append(self._entry_order(p))
        return self._plan_table(plans, now, after.delta_dollars / spx_price), after

    async def _price_trade(
        self, trade: DiscretionaryTradeConfig, spx_price: float
    ) -> PlannedTrade:
        """Pick one unit's contracts and its greeks; a note says why if it can't."""
        plan = PlannedTrade(trade)
        sconfig = self.symbol_config(trade.symbol)
        if sconfig is not None and sconfig.no_trading:
            plan.note = "no_trading is set"
            return plan
        try:
            plan.legs = await self._trade_legs(trade)
        except Exception as e:  # noqa: BLE001
            plan.note = f"skipped: {e}"
            return plan
        beta = await self.beta(trade.symbol)
        if beta is None:
            beta = 1.0
            plan.note = "beta defaulted to 1.0"
        for leg in plan.legs:
            g, size = leg.greeks, leg.sign * float(leg.contract.multiplier or 100)
            plan.unit_share_delta += g.delta * size
            plan.unit_gamma += g.gamma * size * (beta * g.undPrice / spx_price) ** 2
            plan.unit_theta += g.theta * size
            plan.unit_vega += g.vega * size
        plan.unit_plain_dollars = plan.unit_share_delta * plan.legs[0].greeks.undPrice
        plan.unit_beta_dollars = plan.unit_plain_dollars * beta
        plan.unit_spx = plan.unit_beta_dollars / spx_price
        return plan

    def _entry_order(self, plan: PlannedTrade) -> tuple[Contract, LimitOrder]:
        """Single leg: BUY/SELL at mid. Vertical: one BUY combo (negative = credit).
        Tagged ENTRY_ORDER_REF so a rerun cancels it if it is still working."""
        legs = [
            (leg.contract, "BUY" if leg.sign > 0 else "SELL", leg.price)
            for leg in plan.legs
        ]
        return self._priced_order(
            plan.trade.symbol,
            legs,
            plan.units,
            ENTRY_ORDER_REF,
            self.order_ops.get_order_exchange(),
        )

    @staticmethod
    def _plan_table(plans: list[PlannedTrade], now: float, after: float) -> Table:
        table = Table(
            title=f"Discretionary trades: SPX delta {now:+.2f} -> {after:+.2f}",
            caption=(
                "Existing positions stay as they are; only these trades are sized. "
                "Contracts are whole units (spreads: whole spreads), never in the "
                "wrong direction. Orders go out in the post_discretionary stage "
                "(dry run: shown only)."
            ),
            caption_justify="left",
        )
        for column in ("Trade", "Contracts", "SPX Δ / unit", "Weight", "Open", "Note"):
            table.add_column(
                column,
                justify="left" if column in ("Trade", "Contracts", "Note") else "right",
            )
        for p in plans:
            t = p.trade
            table.add_row(
                f"{t.symbol} {t.structure}",
                " / ".join(
                    f"{'+' if leg.sign > 0 else '-'}{leg.contract.localSymbol}"
                    for leg in p.legs
                )
                or "-",
                f"{p.unit_spx:+.3f}" if p.legs else "-",
                f"{t.weight:g}",
                str(p.units),
                p.note,
            )
        return table
