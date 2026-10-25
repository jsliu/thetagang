from collections.abc import Mapping
from typing import Any


def redgreen(value: float) -> str:
    if value >= 0:
        return "green"
    return "red"


def dfmt(amount: str | float | None, precision: int = 2) -> str:
    if amount is not None:
        amount = float(amount)
        rg = redgreen(amount)
        return f"[{rg}]${amount:,.{precision}f}[/{rg}]"
    return ""


def pfmt(amount: str | float | None, precision: int = 2) -> str:
    if amount is None:
        return ""
    amount = float(amount) * 100.0
    rg = redgreen(amount)
    return f"[{rg}]{amount:.{precision}f}%[/{rg}]"


def ffmt(amount: float | None, precision: int = 2) -> str:
    if amount is not None:
        amount = float(amount)
        rg = redgreen(amount)
        return f"[{rg}]{amount:.{precision}f}[/{rg}]"
    return ""


def ifmt(amount: int | None) -> str:
    if amount is not None:
        amount = int(amount)
        rg = redgreen(amount)
        return f"[{rg}]{amount:,d}[/{rg}]"
    return ""


def to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def contract_label(contract: Any, legs: Mapping[int, Any] | None = None) -> str:
    """Short name for a contract, as in the Spreads table: "RDDT 20261030 185C"
    for an option, "RDDT 20261030 185/215C" for a vertical, "SYM EXP a/bR -> EXP
    c/dR" for a roll, the symbol for stock. Combo legs are looked up in `legs`
    (conId -> contract); unknown legs show their conId."""
    sec_type = getattr(contract, "secType", "")
    if sec_type == "OPT":
        right = str(contract.right).upper()[:1]
        return (
            f"{contract.symbol} {contract.lastTradeDateOrContractMonth} "
            f"{contract.strike:g}{right}"
        )
    if sec_type != "BAG":
        return str(getattr(contract, "symbol", "") or contract)
    groups: dict[tuple[str, str], list[float]] = {}
    unknown: list[str] = []
    for leg in getattr(contract, "comboLegs", None) or []:
        leg_contract = (legs or {}).get(leg.conId)
        if leg_contract is None:
            unknown.append(f"conId {leg.conId}")
            continue
        key = (
            leg_contract.lastTradeDateOrContractMonth,
            str(leg_contract.right).upper()[:1],
        )
        groups.setdefault(key, []).append(leg_contract.strike)
    parts = [
        f"{expiry} {'/'.join(f'{k:g}' for k in sorted(strikes))}{right}"
        for (expiry, right), strikes in sorted(groups.items())
    ]
    return f"{contract.symbol} " + " -> ".join(parts + unknown)
