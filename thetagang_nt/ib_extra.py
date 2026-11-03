"""Account values NautilusTrader's IB adapter doesn't expose: cushion, buying
power, excess liquidity, margins, cash, IB's daily P&L per position, and every
open order (from any client: the strategy, pretrade, TWS).

A small read-only ib_async connection (client id = the node's + EXTRA_CLIENT_OFFSET)
inside the monitor process. It never places orders (IB read-only mode) and a
failure only leaves these fields out of the dashboard."""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from ib_async import IB, Contract

EXTRA_CLIENT_OFFSET = 100
SUMMARY_TAGS = {
    "TotalCashValue": "cash",
    "ExcessLiquidity": "excess_liquidity",
    "InitMarginReq": "init_margin",
    "FullMaintMarginReq": "maint_margin",
    "BuyingPower": "buying_power",
    "Cushion": "cushion",
}


DONE = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def _number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and abs(x) < 1e300 else None  # IB's unset: 1.8e308


def contract_label(contract: Any) -> str:
    """The dashboard's name for a stock or option: "AA" or "AA 20270115 40P"."""
    if contract.secType in ("OPT", "FOP"):
        expiry = contract.lastTradeDateOrContractMonth
        return f"{contract.symbol} {expiry} {contract.strike:g}{contract.right[:1]}"
    return position_name(contract)


def position_name(contract: Any) -> str:
    """The name Nautilus gives the instrument: IB's local symbol, with spaces
    replaced by '-' for stocks (BRK B -> BRK-B)."""
    name = contract.localSymbol or contract.symbol
    return name.replace(" ", "-") if contract.secType == "STK" else name


class AccountExtras:
    def __init__(
        self, host: str, port: int, client_id: int, account: str | None
    ) -> None:
        self.host = os.environ.get("IB_HOST", host)  # docker: host.docker.internal
        self.port, self.client_id, self.account = port, client_id, account
        self.ib = IB()
        self._pnl: dict[int, Any] = {}  # conId -> live PnLSingle subscription
        self._account_pnl: Any = None  # live PnL subscription for the account
        self._labels: dict[int, str] = {}  # conId -> contract_label, combo legs
        self._lock = asyncio.Lock()  # --role all: the monitor and strategy share it

    async def connect(self) -> None:
        async with self._lock:
            if not self.ib.isConnected():
                await self.ib.connectAsync(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    readonly=True,
                    timeout=15,
                )
                self._pnl, self._account_pnl = {}, None

    async def read(self) -> tuple[dict[str, float], dict[str, float]]:
        """(account fields, IB daily P&L by position name)."""
        await self.connect()
        account = self.account or self.ib.managedAccounts()[0]
        summary: dict[str, float] = {}
        for v in await self.ib.accountSummaryAsync(account):
            field = SUMMARY_TAGS.get(v.tag)
            if (
                field
                and v.currency in ("USD", "BASE", "")
                and (x := _number(v.value)) is not None
            ):
                summary[field] = x
        # the account's daily P&L as TWS shows it: open positions plus those closed
        # today and commissions, which a sum over open positions can't include
        if self._account_pnl is None:
            self._account_pnl = self.ib.reqPnL(account, "")
        for field, value in (
            ("daily_pnl", self._account_pnl.dailyPnL),
            ("realized_pnl", self._account_pnl.realizedPnL),  # TWS's Realized P&L
            ("unrealized_pnl", self._account_pnl.unrealizedPnL),  # TWS's Unrealized
        ):
            if (x := _number(value)) is not None:
                summary[field] = x
        held = {
            p.contract.conId: p.contract
            for p in self.ib.positions(account)
            if p.position
        }
        for con_id in set(self._pnl) - set(held):
            self.ib.cancelPnLSingle(account, "", con_id)
            del self._pnl[con_id]
        for con_id in set(held) - set(self._pnl):
            self._pnl[con_id] = self.ib.reqPnLSingle(account, "", con_id)
        daily = {
            position_name(held[con_id]): x
            for con_id, pnl in self._pnl.items()
            if (x := _number(pnl.dailyPnL)) is not None
        }
        return summary, daily

    async def open_orders(self) -> list[dict[str, Any]]:
        """Every working order at IB, from any client: what it trades, type,
        limit, quantity, filled, status."""
        await self.connect()
        out = []
        for t in await self.ib.reqAllOpenOrdersAsync():
            c, o, s = t.contract, t.order, t.orderStatus
            if s.status in DONE:
                continue
            if c.secType == "BAG":  # a combo: each leg's side, flipped on a sell
                await self._learn(leg.conId for leg in c.comboLegs)
                flip = {"BUY": "SELL", "SELL": "BUY"}
                text = " / ".join(
                    f"{leg.action if o.action == 'BUY' else flip[leg.action]}"
                    f" {self._labels.get(leg.conId, leg.conId)}"
                    for leg in c.comboLegs
                )
            else:
                text = f"{o.action} {contract_label(c)}"
            out.append(
                {
                    "order": text,
                    "symbol": c.symbol.replace(" ", "-"),
                    "type": o.orderType,
                    "status": s.status,
                    "client": str(o.clientId),
                    "id": str(o.permId or o.orderId),
                    "qty": _number(o.totalQuantity),
                    "limit": _number(o.lmtPrice) if "LMT" in o.orderType else None,
                    "filled": _number(s.filled),
                }
            )
        return out

    async def order_symbols(self) -> set[str]:
        """Underlying symbols with a working order at IB, from any client, as the
        strategy names them (index roots also as "^VIX")."""
        symbols = {o["symbol"] for o in await self.open_orders()}
        return symbols | {f"^{s}" for s in symbols}

    async def _learn(self, con_ids: Any) -> None:
        unknown = [c for c in con_ids if c not in self._labels]
        if unknown:
            for c in await self.ib.qualifyContractsAsync(
                *[Contract(conId=i) for i in unknown]
            ):
                if c is not None:
                    self._labels[c.conId] = contract_label(c)

    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
