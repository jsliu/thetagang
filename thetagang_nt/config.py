"""Native Nautilus configs, loaded from one TOML file (see thetagang.nt.toml)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

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
    delta: float | None = None  # credit.roll_target delta override
    max_dte: int | None = None  # credit.roll_target max_dte override
    # options here with no quote (so no greeks) count as zero delta when sizing
    # the trade plan instead of blocking it; the dashboard still flags them
    ignore_missing_greeks: bool = False


class TradingConfig(NautilusConfig, frozen=True):
    """[strategy.trading]: when and how the trading process acts."""

    dry_run: bool = True  # log orders instead of sending them
    interval_secs: int = 300
    market_hours_only: bool = True
    # the trade plan resizes what is held toward the positions summary's SPX
    # delta 0 targets (shares, one option, or a whole vertical per symbol).
    # "auto": place it when off SPX delta neutral; "manual": only show it, and
    # trade with `thetagang-nt pretrade`
    approve_trades: str = "auto"
    proposal_max_age_secs: int = 300  # a confirmation later than this sends nothing
    reprice_after_secs: int | None = 60  # move unfilled orders halfway to mid, once
    # --role trade only: the trading process's own metrics (orders, dry_run, trade plan)
    metrics_port: int = 9465
    pretrade_metrics_port: int = 9466  # the pre-trade dashboard, while it asks

    def __post_init__(self) -> None:
        if self.approve_trades not in ("auto", "manual"):
            raise ValueError(
                f"approve_trades must be auto or manual, not {self.approve_trades!r}"
            )


class DeltaBandConfig(NautilusConfig, frozen=True):
    """SPX delta neutral (no new trades) while |beta delta $| <= the band: a 1%
    SPX day may cost at most theta_days days of theta, capped at nav_pct of net
    liq (rules.delta_band_dollars)."""

    theta_days: float = 2.0
    nav_pct: float = 0.10


class ChainConfig(NautilusConfig, frozen=True):
    """How a contract is picked by delta from the option chain."""

    strikes: int = 15
    expirations: int = 2
    quote_timeout_secs: float = 10.0
    minimum_credit: float = 0.0


class CreditConfig(NautilusConfig, frozen=True):
    """[strategy.credit]: standalone short calls/puts and credit spreads."""

    roll_target: TargetConfig = TargetConfig()
    roll_when: RollWhenConfig = RollWhenConfig()


class DebitRollWhenConfig(NautilusConfig, frozen=True):
    """Roll a debit spread or long option once dte <= dte or its long leg's
    |delta| >= delta (unless take-profit closed it)."""

    dte: int = 10
    delta: float = 0.0  # also roll once the long leg's |delta| >= this; 0: off
    max_dte: int | None = None
    close_if_unable_to_roll: bool = False

    def rules(self) -> RollWhen:
        return RollWhen(
            dte=self.dte,
            max_dte=self.max_dte,
            close_if_unable_to_roll=self.close_if_unable_to_roll,
        )


class DebitConfig(NautilusConfig, frozen=True):
    """[strategy.debit]: debit spreads and standalone long calls/puts."""

    take_profit: float = 0.8  # spreads: share of the max profit (width - debit)
    long_take_profit_gain: float = 1.0  # longs: close at (1 + this) x premium paid
    max_roll_debit: float = 0.5  # x width (spreads) or x old option value (longs)
    roll_when: DebitRollWhenConfig = DebitRollWhenConfig()
    roll_target: TargetConfig = TargetConfig()


class ButterflyConfig(NautilusConfig, frozen=True):
    """[strategy.butterfly]: what the Structures table recommends."""

    trade: bool = False  # true: the bot places these recommendations (dry_run applies)
    take_profit: float = 0.5  # close at this share of the max profit
    partial_take_profit: float = 0.25  # close half (2+ held) at this share
    close_dte: int = 7  # close with this many days or fewer left


class StraddleConfig(NautilusConfig, frozen=True):
    """[strategy.straddle]: long straddles and strangles."""

    trade: bool = False  # true: the bot places these recommendations (dry_run applies)
    take_profit_gain: float = 0.5  # close at P&L >= this x the debit
    partial_take_profit_gain: float = 0.25  # close half (2+ held)
    leg_delta: float = 0.7  # a winning leg at |delta| >= this: close that leg
    roll_dte: int = 21  # roll the whole position out in time


class CalendarConfig(NautilusConfig, frozen=True):
    """[strategy.calendar]: calendars and diagonals."""

    trade: bool = False  # true: the bot places these recommendations (dry_run applies)
    take_profit: float = 0.25  # close at P&L >= this x the debit
    short_roll_dte: int = 5  # roll the short leg out with this many days left
    close_dte: int = 10  # close once the long leg has this many days left


class StructureGroupConfig(NautilusConfig, frozen=True):
    """[[strategy.structures]]: option legs you hold as one position, by the
    dashboard's contract labels ("AA 20261218 40P"); left to you ("by hand")."""

    name: str
    legs: list[str]


class DiscretionaryConfig(StrategyConfig, frozen=True):
    trading: TradingConfig = TradingConfig()
    delta_band: DeltaBandConfig = DeltaBandConfig()
    chain: ChainConfig = ChainConfig()
    credit: CreditConfig = CreditConfig()
    debit: DebitConfig = DebitConfig()
    vix_call_hedge: VixHedgeConfig = VixHedgeConfig()
    symbols: dict[str, SymbolConfig] = {}
    butterfly: ButterflyConfig = ButterflyConfig()
    straddle: StraddleConfig = StraddleConfig()
    calendar: CalendarConfig = CalendarConfig()
    structures: list[StructureGroupConfig] = []

    def groups(self) -> dict[str, tuple[str, ...]]:
        return {g.name: tuple(g.legs) for g in self.structures}


class MonitorConfig(ActorConfig, frozen=True):
    interval_secs: int = 60
    metrics_addr: str = "0.0.0.0"
    metrics_port: int = 9464
    publish: bool = (
        True  # serve /metrics; off in --role trade, which only needs the betas
    )
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
    # compare positions with IB this often, so trades made elsewhere (TWS, the
    # phone app) reach the cache without a restart; 0: only at startup
    position_check_secs: int = 60


class CandidateConfig(NautilusConfig, frozen=True):
    """One idea for `thetagang-nt pretrade` to compare: its trades are sized
    together to bring SPX delta to 0."""

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


class PMStrategyConfig(NautilusConfig, frozen=True):
    """[[pm.strategies]]: your inputs for one strategy, by the ID the Portfolio
    Manager dashboard shows (e.g. "AA-debit-spread-1"). Copied into the journal."""

    id: str
    thesis: str = ""
    invalidation_below: float | None = None  # thesis invalid if the price falls below
    invalidation_above: float | None = None  # ...or rises above
    target: float | None = None  # profit target, $
    max_loss: float | None = None  # max acceptable loss (risk budget), $
    thesis_valid: bool | None = None
    # short premium: letting it expire or be assigned is the plan (a cash-secured
    # put, a covered call), so no assignment review [S4]
    assignment_intended: bool = False


class PMConfig(NautilusConfig, frozen=True):
    """[pm]: the Portfolio Manager (decision support, vault PM Knowledge Base)."""

    journal: bool = True  # write the journal to Postgres at $JOURNAL_DSN (if set)
    # the vault's rules win: the bot's roll/close rules no longer place orders;
    # the PM's review signals tell you what to look at, you decide. The trade
    # plan and the VIX hedge are unaffected. false: the bot's rules trade again
    decide: bool = True
    # risk budget, $, for every strategy without its own [[pm.strategies]]
    # max_loss: a P0 CLOSE review when the worst case exceeds it; unset: off
    max_loss: float | None = None
    strategies: list[PMStrategyConfig] = []
    # review thresholds, by vault register ID; all hypotheses (assumptions_register)
    profit_levels: tuple[float, ...] = (0.5, 1.0, 1.5)  # [T1] return on debit
    spread_max_profit: float = 0.8  # [T1] defined-risk debit: share of max profit
    # [T2] superseded by S1: credit strategies go to the Short Options Manager
    credit_levels: tuple[float, ...] = (0.5, 0.75)
    delta_review: float = 0.8  # [T4] a single long option's |delta|
    delta_drift: float | None = None  # [T5] share of entry gross delta; unset: off
    gt_review: float = 1.0  # [T6] long convexity: GT_h below this
    gt_horizon: int = 5  # [T6] trading days
    remaining_reward_risk: float = 0.25  # [T8] below this
    near_expiry_days: int = 7  # a short leg this close to expiry...
    near_money: float = 0.02  # ...and within this of its strike
    stale_secs: float = 300.0  # [T9] quote older than this
    wide_spread: float = 0.25  # [T9] (ask - bid) / mid above this
    # [T3] (original life, review at days left), interpolated between rows
    dte_windows: tuple[tuple[int, int], ...] = (
        (90, 28),
        (180, 60),
        (365, 120),
        (730, 270),
    )
    gamma_tolerance: float = (
        0.01  # [T7] |net gamma| below this counts as zero (shares per $1)
    )
    theta_tolerance: float = (
        0.01  # [T7] |net theta| below this counts as zero ($ per day)
    )
    grid_moves: tuple[float, ...] = (
        -0.05,
        -0.02,
        0.0,
        0.02,
        0.05,
    )  # [M1] gamma sign check
    realized_vol_days: int = 21  # [M6] daily returns behind the realised-vol forecast
    # Short Options Manager (strategies opened for a credit), short_exit_signals
    capture_levels: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9)  # [S1] credit captured
    # [S2] front short leg's DTE at or below: review / prefer close or adjust / close
    short_dte_bands: tuple[int, int, int] = (21, 14, 7)
    loss_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)  # [S3] loss / credit
    assignment_days: int = 7  # [S4] an in-the-money short leg this close to expiry
    tested_moves: float = 1.0  # [S5] spot within this many moves of a short strike
    tested_horizon: int = 5  # [S5] trading days in a move (S * sigma_f * sqrt(h/252))
    # Long VIX calls (vault vix_options_management): replace T1 and T3 for VIX
    vix_profit_levels: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30)  # [V2] on the debit
    vix_loss_levels: tuple[float, ...] = (-0.25, -0.35)  # [V3] loss reviews, not stops
    vix_review_dte: int = 14  # [V7] review at this many days left (starting point)

    def limits(self) -> Any:
        from thetagang_nt.pm.long_options.signals import SignalLimits

        return SignalLimits(
            profit_levels=tuple(self.profit_levels),
            spread_max_profit=self.spread_max_profit,
            credit_levels=tuple(self.credit_levels),
            delta_review=self.delta_review,
            delta_drift=self.delta_drift,
            gt_review=self.gt_review,
            gt_horizon=self.gt_horizon,
            remaining_reward_risk=self.remaining_reward_risk,
            near_expiry_days=self.near_expiry_days,
            near_money=self.near_money,
            dte_windows=self.dte_windows,
        )

    def short_limits(self) -> Any:
        from thetagang_nt.pm.short_options.signals import ShortLimits

        return ShortLimits(
            capture_levels=tuple(self.capture_levels),
            dte_bands=tuple(self.short_dte_bands),
            loss_multiples=tuple(self.loss_multiples),
            assignment_days=self.assignment_days,
            tested_moves=self.tested_moves,
            tested_horizon=self.tested_horizon,
        )

    def vix_limits(self) -> Any:
        from thetagang_nt.pm.long_options.vix import VixLimits

        return VixLimits(
            profit_levels=tuple(self.vix_profit_levels),
            loss_levels=tuple(self.vix_loss_levels),
            review_dte=self.vix_review_dte,
        )

    def quality(self) -> Any:
        from thetagang_nt.pm.core.quality import QualityLimits

        return QualityLimits(stale_secs=self.stale_secs, wide_spread=self.wide_spread)

    def tolerance(self) -> Any:
        from thetagang_nt.pm.core.classify import Tolerance

        return Tolerance(gamma=self.gamma_tolerance, theta=self.theta_tolerance)

    def params(self) -> list[tuple[str, str, str, str, str]]:
        """(name, value, vault ID, default | your setting, meaning) for every
        setting the Portfolio Manager decides with: the dashboard shows them."""
        default = PMConfig()

        def show(v: Any) -> str:
            if isinstance(v, tuple):
                return (
                    ", ".join(show(x) for x in v)
                    if v and not isinstance(v[0], tuple)
                    else "; ".join(f"{a}d -> {b}d" for a, b in v)
                )
            return (
                "off"
                if v is None
                else f"{v:g}"
                if isinstance(v, float)
                else str(v).lower()
            )

        rows = (
            (
                "decide",
                "",
                "the PM's review signals replace the bot's roll/close orders",
            ),
            (
                "max_loss",
                "",
                "risk budget, $: P0 when a strategy's worst case exceeds it",
            ),
            ("profit_levels", "T1", "debit: review at these gains on the debit"),
            (
                "spread_max_profit",
                "T1",
                "defined-risk debit: review at this share of max profit",
            ),
            (
                "credit_levels",
                "T2",
                "superseded by S1 (credit strategies: Short Options Manager)",
            ),
            ("dte_windows", "T3", "original life -> review when this many days remain"),
            ("delta_review", "T4", "a single long option above this |delta|"),
            ("delta_drift", "T5", "net delta moved by this share of entry gross delta"),
            ("gt_review", "T6", "long convexity: GT_h below this (decay beats gamma)"),
            ("gt_horizon", "T6", "trading days for GT_h"),
            ("gamma_tolerance", "T7", "|net gamma| below this counts as zero"),
            ("theta_tolerance", "T7", "|net theta| below this counts as zero"),
            ("remaining_reward_risk", "T8", "remaining reward / risk below this"),
            ("stale_secs", "T9", "a quote older than this many seconds is stale"),
            ("wide_spread", "T9", "(ask - bid) / mid above this is wide"),
            ("near_expiry_days", "", "a short leg this close to expiry..."),
            ("near_money", "", "...and this close to its strike"),
            ("grid_moves", "M1", "price moves for the gamma sign check"),
            (
                "realized_vol_days",
                "M6",
                "daily returns behind the realised-vol forecast",
            ),
            ("capture_levels", "S1", "short: review at these shares of the credit"),
            (
                "short_dte_bands",
                "S2",
                "short: front short leg DTE review / adjust / close bands",
            ),
            ("loss_multiples", "S3", "short: review at these losses per $ of credit"),
            (
                "assignment_days",
                "S4",
                "short: an in-the-money short leg this close to expiry",
            ),
            ("tested_moves", "S5", "short: spot within this many moves of a strike"),
            ("tested_horizon", "S5", "short: trading days in one move"),
            ("vix_profit_levels", "V2", "long VIX: review at these gains on the debit"),
            ("vix_loss_levels", "V3", "long VIX: loss review at these returns"),
            ("vix_review_dte", "V7", "long VIX: review when this many days remain"),
            ("journal", "", "write the journal (Postgres)"),
        )
        return [
            (
                name,
                show(getattr(self, name)),
                vid,
                "default"
                if getattr(self, name) == getattr(default, name)
                else "your setting",
                meaning,
            )
            for name, vid, meaning in rows
        ]


class AppConfig(NautilusConfig, frozen=True):
    node: NodeConfig = NodeConfig()
    ib: IBConfig = IBConfig()
    monitor: MonitorConfig = MonitorConfig()
    strategy: DiscretionaryConfig | None = None  # omit for monitor-only
    pretrade: PreTradeSectionConfig = PreTradeSectionConfig()
    pm: PMConfig = PMConfig()


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
      on strategy.trading.metrics_port.
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
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            allow_overfills=True,
            position_check_interval_secs=app.node.position_check_secs or None,
        ),
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
