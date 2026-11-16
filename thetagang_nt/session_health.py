"""IB session health telemetry for the active Nautilus node."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class IBSessionHealth:
    def __init__(
        self,
        role: str,
        client_id: int,
        configured_account: str | None,
        connection_check: Callable[[], bool],
    ) -> None:
        from prometheus_client import Gauge

        self._role = role
        self._client_id = str(client_id)
        self._configured_account = configured_account
        self._connection_check = connection_check
        self._gauge = Gauge(
            "thetagang_ib_session_healthy",
            "1 while this process's IB session is connected to its configured account",
            ["role", "client_id", "account"],
        )
        self.update(False, None)

    def refresh(self, observed_account: object | None) -> None:
        try:
            connected = self._connection_check()
        except Exception:  # noqa: BLE001 - health telemetry must fail closed
            connected = False
        self.update(connected, observed_account)

    def refresh_from_cache(self, cache: Any, connected: bool | None = None) -> None:
        try:
            accounts = [
                account.id
                for account in cache.accounts()
                if account.id.get_issuer() == "IB"
            ]
            observed_account = accounts[0] if len(accounts) == 1 else None
        except Exception:  # noqa: BLE001 - health telemetry must fail closed
            observed_account = None

        if connected is None:
            self.refresh(observed_account)
        else:
            self.update(connected, observed_account)

    def update(self, connected: bool, observed_account: object | None) -> None:
        account_id = getattr(observed_account, "get_id", None)
        if callable(account_id):
            account = str(account_id())
        else:
            account = "" if observed_account is None else str(observed_account)

        self._gauge.clear()
        self._gauge.labels(
            role=self._role,
            client_id=self._client_id,
            account=account,
        ).set(
            int(
                connected
                and bool(self._configured_account)
                and account == self._configured_account
            )
        )
