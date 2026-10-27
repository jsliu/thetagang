"""Native Nautilus configs, loaded from one TOML file (see thetagang.nt.toml)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import msgspec
from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.config import (
    DockerizedIBGatewayConfig,
    IBMarketDataTypeEnum,
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)
from nautilus_trader.config import (
    ActorConfig,
    CacheConfig,
    DatabaseConfig,
    LiveExecEngineConfig,
    LoggingConfig,
    NautilusConfig,
    StrategyConfig,
    TradingNodeConfig,
)

from thetagang_nt.rules import STRUCTURES, RollWhen


class RightRules(NautilusConfig, frozen=True):
    itm: bool = True  # roll in-the-money options
    always_when_itm: bool = False  # roll ITM regardless of P&L
    credit_only: bool = False  # only roll for a net credit >= minimum_credit


class RollWhenConfig(NautilusConfig, frozen=True):
    """Roll a short option (or credit vertical) when dte <= dte and P&L >= min_pnl,
    or P&L >= pnl (within max_dte). Close when P&L > close_at_pnl."""

    pnl: float = 0.9
    dte: int = 10
    min_pnl: float = 0.0
    max_dte: int | None = None
    close_at_pnl: float = 1.0
    close_if_unable_to_roll: bool = False
    calls: RightRules = RightRules()
    puts: RightRules = RightRules()

    def rules(self) -> RollWhen:
        return RollWhen(
            pnl=self.pnl,
            dte=self.dte,
            min_pnl=self.min_pnl,
            max_dte=self.max_dte,
            close_at_pnl=self.close_at_pnl,
            close_if_unable_to_roll=self.close_if_unable_to_roll,
            itm={"C": self.calls.itm, "P": self.puts.itm},
            always_when_itm={
                "C": self.calls.always_when_itm,
                "P": self.puts.always_when_itm,
            },
            credit_only={"C": self.calls.credit_only, "P": self.puts.credit_only},
        )


class TargetConfig(NautilusConfig, frozen=True):
    """What a roll moves into."""

    dte: int = 30
    max_dte: int | None = None
    delta: float = 0.3


class TradeConfig(NautilusConfig, frozen=True):
    """A trade opened to bring SPX delta to 0 (see rules.STRUCTURES)."""

    symbol: str
    structure: str
    delta: float = 0.3
    dte: int = 30
    width: float | None = None
    weight: float = 1.0
    max_contracts: int | None = None

    def __post_init__(self) -> None:
        if self.structure not in STRUCTURES:
            raise ValueError(f"{self.symbol}: unknown structure {self.structure!r}")
        if (STRUCTURES[self.structure][2] is not None) != (self.width is not None):
            raise ValueError(
                f"{self.symbol} {self.structure}: width is required for spreads "
                "and only for them"
            )


class AllocationConfig(NautilusConfig, frozen=True):
    lower_bound: float | None = None
    upper_bound: float | None = None
    weight: float = 0.0


class VixHedgeConfig(NautilusConfig, frozen=True):
    """Buy VIX calls per VIXMO allocation; sell them all above the threshold."""

    enabled: bool = False
    delta: float = 0.3
    target_dte: int = 30
    close_hedges_when_vix_exceeds: float | None = None
    ignore_dte: int = 0
    allocation: list[AllocationConfig] = []


class SymbolConfig(NautilusConfig, frozen=True):
    no_trading: bool = False
    put_strike_limit: float | None = None  # never roll a put above this strike
    call_strike_limit: float | None = None  # never roll a call below this strike
    delta: float | None = None  # roll target delta override
    max_dte: int | None = None  # roll target max_dte override
    # options here with no quote (so no greeks) count as zero delta when sizing
    # the trade plan instead of blocking it; the dashboard still flags them
    ignore_missing_greeks: bool = False


class DiscretionaryConfig(StrategyConfig, frozen=True):
    dry_run: bool = True  # log orders instead of sending them
    interval_secs: int = 300
    market_hours_only: bool = True
    # SPX delta neutral (no new trades) while |beta delta $| <= the band: a 1%
    # SPX day may cost at most delta_band_theta_days days of theta, capped at
    # delta_band_nav_pct of net liq (rules.delta_band_dollars)
    delta_band_theta_days: float = 2.0
    delta_band_nav_pct: float = 0.10
    minimum_credit: float = 0.0
    debit_spread_take_profit: float = 0.8
    debit_spread_max_roll_debit: float = 0.5
    reprice_after_secs: int | None = 60  # move unfilled orders halfway to mid, once
    chain_strikes: int = 15
    chain_expirations: int = 2
    quote_timeout_secs: float = 10.0
    roll_target: TargetConfig = TargetConfig()
    roll_when: RollWhenConfig = RollWhenConfig()
    trades: list[TradeConfig] = []
    vix_call_hedge: VixHedgeConfig = VixHedgeConfig()
    symbols: dict[str, SymbolConfig] = {}
    # --role trade only: the trading process's own metrics (orders, dry_run, trade plan)
    metrics_port: int = 9465
    # "auto": open the [[strategy.trades]] when off SPX delta neutral; "manual":
    # only show the plan, and open trades with `thetagang-nt pretrade`
    approve_trades: str = "auto"
    # the trade plan's way to SPX delta 0: false = open the [[strategy.trades]];
    # true = resize what is held toward the positions summary's targets (shares,
    # one option, or a whole vertical per symbol). One or the other, not both.
    place_target_orders: bool = False
    pretrade_metrics_port: int = 9466  # the pre-trade dashboard, while it asks
    proposal_max_age_secs: int = 300  # a confirmation later than this sends nothing

    def __post_init__(self) -> None:
        if self.approve_trades not in ("auto", "manual"):
            raise ValueError(
                f"approve_trades must be auto or manual, not {self.approve_trades!r}"
            )
        if self.place_target_orders and self.trades:
            raise ValueError(
                "use either [[strategy.trades]] or place_target_orders, not both: "
                "each hedges the whole SPX delta on its own"
            )


class MonitorConfig(ActorConfig, frozen=True):
    interval_secs: int = 60
    metrics_addr: str = "0.0.0.0"
    metrics_port: int = 9464
    publish: bool = (
        True  # serve /metrics; off in --role trade, which only needs the betas
    )
    long_option_alert_delta: float = 0.8
    stress_moves: tuple[float, ...] = (-0.10, -0.05, -0.01, 0.01, 0.05, 0.10)


class IBConfig(NautilusConfig, frozen=True):
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 11
    account_id: str | None = None  # or the TWS_ACCOUNT env var
    market_data_type: str = "REALTIME"  # REALTIME, DELAYED, DELAYED_FROZEN, FROZEN
    gateway: DockerizedIBGatewayConfig | None = None  # run IB Gateway in docker


class NodeConfig(NautilusConfig, frozen=True):
    trader_id: str = "THETAGANG-001"
    log_level: str = "INFO"
    log_directory: str | None = None
    redis_host: str | None = None  # persist the cache in Redis when set
    redis_port: int = 6379


class CandidateConfig(NautilusConfig, frozen=True):
    """One idea for `thetagang-nt pretrade` to compare: its trades are sized
    together to bring SPX delta to 0, like the [[strategy.trades]]."""

    name: str
    trades: list[TradeConfig] = []

    def __post_init__(self) -> None:
        if not self.trades:
            raise ValueError(f"pretrade candidate {self.name!r} has no trades")


class PreTradeSectionConfig(NautilusConfig, frozen=True):
    """[pretrade]: the candidates `thetagang-nt pretrade` compares, each sized on
    its own; you pick the one to send."""

    candidates: list[CandidateConfig] = []
    # the trading strategy compares them in the background this often (market
    # hours), so the Pre-trade dashboard always shows them; 0: only on demand
    refresh_mins: int = 15

    def __post_init__(self) -> None:
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError("pretrade candidates need distinct names")


class AppConfig(NautilusConfig, frozen=True):
    node: NodeConfig = NodeConfig()
    ib: IBConfig = IBConfig()
    monitor: MonitorConfig = MonitorConfig()
    strategy: DiscretionaryConfig | None = None  # omit for monitor-only
    pretrade: PreTradeSectionConfig = PreTradeSectionConfig()


def load(path: str | Path) -> AppConfig:
    with open(path, "rb") as f:
        return msgspec.convert(tomllib.load(f), AppConfig)


ROLES = ("all", "monitor", "trade")


def split(
    app: AppConfig, role: str
) -> tuple[MonitorConfig, DiscretionaryConfig | None, int]:
    """What one process runs: (monitor, strategy or None, IB client id).

    all: monitor + strategy in one process (metrics on monitor.metrics_port).
    monitor: the portfolio only, no strategy, so it can never trade; connects as
      client_id + 1 so it runs beside a trade process.
    trade: the strategy, with a monitor that only loads the betas it sizes with
      (no metrics, so the dashboard isn't doubled); the strategy's own metrics go
      on strategy.metrics_port.
    pretrade (`thetagang-nt pretrade`): like trade, as client_id + 2 so it runs
      beside both."""
    if role == "all":
        return app.monitor, app.strategy, app.ib.client_id
    if role == "monitor":
        return app.monitor, None, app.ib.client_id + 1
    if role in ("trade", "pretrade"):
        if app.strategy is None:
            raise ValueError(f"{role} needs a [strategy] section")
        quiet = msgspec.structs.replace(app.monitor, publish=False)
        offset = 2 if role == "pretrade" else 0
        return quiet, app.strategy, app.ib.client_id + offset
    raise ValueError(f"role must be one of {ROLES}, got {role!r}")


def node_config(app: AppConfig, client_id: int | None = None) -> TradingNodeConfig:
    ib = app.ib
    # one venue for everything, so options find their underlying for greeks
    provider = InteractiveBrokersInstrumentProviderConfig(
        symbol_to_mic_venue={"": "SMART"}
    )
    common = dict(
        ibg_host=os.environ.get("IB_HOST", ib.host),  # docker: host.docker.internal
        ibg_port=ib.port,
        ibg_client_id=ib.client_id if client_id is None else client_id,
        dockerized_gateway=ib.gateway,
        instrument_provider=provider,
    )
    cache = None
    if app.node.redis_host:
        cache = CacheConfig(
            database=DatabaseConfig(host=app.node.redis_host, port=app.node.redis_port)
        )
    return TradingNodeConfig(
        trader_id=app.node.trader_id,
        logging=LoggingConfig(
            log_level=app.node.log_level,
            log_directory=app.node.log_directory,
            log_level_file=app.node.log_level if app.node.log_directory else None,
        ),
        cache=cache,
        # IB reports whole shares for a fractional position (17.738 -> 18): accept
        # its fill instead of leaving a stand-in order open that blocks the symbol
        exec_engine=LiveExecEngineConfig(reconciliation=True, allow_overfills=True),
        data_clients={
            IB: InteractiveBrokersDataClientConfig(
                market_data_type=getattr(IBMarketDataTypeEnum, ib.market_data_type),
                **common,
            )
        },
        exec_clients={
            IB: InteractiveBrokersExecClientConfig(
                account_id=ib.account_id or os.environ.get("TWS_ACCOUNT"), **common
            )
        },
        timeout_connection=90,
        timeout_reconciliation=30,
    )
