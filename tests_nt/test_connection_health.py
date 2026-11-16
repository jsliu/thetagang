import sys
from types import SimpleNamespace

from thetagang_nt.session_health import IBSessionHealth


class FakeGauge:
    def __init__(self, _name, _description, _labels):
        self.samples = {}

    def clear(self):
        self.samples.clear()

    def labels(self, **labels):
        key = tuple(sorted(labels.items()))
        return SimpleNamespace(set=lambda value: self.samples.__setitem__(key, value))


def health(
    monkeypatch,
    role="trade",
    client_id=42,
    account="DU123",
    connection_check=lambda: True,
):
    gauges = []

    def gauge(*args):
        result = FakeGauge(*args)
        gauges.append(result)
        return result

    monkeypatch.setitem(
        sys.modules,
        "prometheus_client",
        SimpleNamespace(Gauge=gauge),
    )
    instance = IBSessionHealth(role, client_id, account, connection_check)
    return instance, gauges[0]


def cache(account_id, issuer="IB"):
    account = SimpleNamespace(
        get_issuer=lambda: issuer,
        get_id=lambda: account_id,
    )
    return SimpleNamespace(accounts=lambda: [SimpleNamespace(id=account)])


def test_session_health_starts_unhealthy_with_effective_identity(monkeypatch):
    _health, metric = health(monkeypatch, role="monitor", client_id=12)

    assert metric.samples == {
        (("account", ""), ("client_id", "12"), ("role", "monitor")): 0
    }


def test_session_health_requires_connected_matching_account(monkeypatch):
    session_health, metric = health(monkeypatch)

    session_health.refresh_from_cache(cache("DU123"))

    assert metric.samples == {
        (("account", "DU123"), ("client_id", "42"), ("role", "trade")): 1
    }


def test_session_health_returns_unhealthy_after_disconnect(monkeypatch):
    connected = True
    session_health, metric = health(monkeypatch, connection_check=lambda: connected)
    session_health.refresh_from_cache(cache("DU123"))

    connected = False
    session_health.refresh_from_cache(cache("DU123"))

    assert metric.samples == {
        (("account", "DU123"), ("client_id", "42"), ("role", "trade")): 0
    }


def test_session_health_rejects_mismatched_account(monkeypatch):
    session_health, metric = health(monkeypatch)

    session_health.update(True, "DU999")

    assert metric.samples == {
        (("account", "DU999"), ("client_id", "42"), ("role", "trade")): 0
    }
