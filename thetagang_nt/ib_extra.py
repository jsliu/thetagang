"""Account values NautilusTrader's IB adapter doesn't expose: cushion, buying
power, excess liquidity, margins, cash, and IB's daily P&L per position.

A small read-only ib_async connection (client id = the node's + EXTRA_CLIENT_OFFSET)
inside the monitor process. It never places orders (IB read-only mode) and a
failure only leaves these fields out of the dashboard."""

from __future__ import annotations

import math
import os
from typing import Any

from ib_async import IB

EXTRA_CLIENT_OFFSET = 100
SUMMARY_TAGS = {
    "TotalCashValue": "cash",
    "ExcessLiquidity": "excess_liquidity",
    "InitMarginReq": "init_margin",
    "FullMaintMarginReq": "maint_margin",
    "BuyingPower": "buying_power",
    "Cushion": "cushion",
}


def _number(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


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

    async def read(self) -> tuple[dict[str, float], dict[str, float]]:
        """(account fields, IB daily P&L by position name)."""
        if not self.ib.isConnected():
            await self.ib.connectAsync(
                self.host, self.port, clientId=self.client_id, readonly=True, timeout=15
            )
            self._pnl = {}
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

    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
