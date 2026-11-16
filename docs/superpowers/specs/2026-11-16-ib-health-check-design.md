# IB connection health check

## Intent

Do not report thetagang as healthy unless its IB session is connected to the
expected account with the expected process role and client ID. This is for the
single-operator Docker setup that exposes the monitor and trading metrics on
localhost.

## Goals

- Publish live connection state and actual role, account, and client ID for
  monitor and trade processes.
- Compare those values with an independent expected account and base client ID.
- Check health once after monitoring starts, expose a command for manual checks,
  and alert periodically through the existing Grafana/Telegram monitoring.
- Fail unhealthy when data is missing, stale, disconnected, or mismatched.

## Non-goals

- Do not stop or change trading based on monitoring health.
- Do not add a direct IB connection from the monitoring script.
- Do not change the separate intraday monitoring checks.

## Approaches

1. **Runtime identity metrics and monitoring checks (chosen).** The bot reports
   its configured role/client ID and the account observed in its live IB
   session, with connection state. Monitoring compares these against expected
   values and alerts. This verifies the process that is actually running.
2. **Inspect Docker commands and existing account metrics only.** This avoids
   bot changes, but proves only what Docker was asked to start, not whether that
   IB client connected.
3. **Probe IB from `run.sh`.** This verifies that IB accepts a separate probe,
   not that the bot's own client is connected, and risks duplicating its client
   ID or testing a different session.

## Design

- Add a low-cardinality Prometheus health metric for each bot role. It reports
  the role, effective IB client ID, observed account, and whether that process's
  IB session is connected and matches its configured account.
- Add expected account ID and base client ID to the monitoring environment.
  The monitor role is expected at base ID + 1; the trade role uses the base ID.
  The live trade role is required only when its service is running.
- Add a `run.sh health` check that queries the bot metrics and prints healthy
  only when required identities match and their connection signals are current.
  Run it after `run.sh up` has started Prometheus. Use the existing periodic
  Grafana/Telegram alerting to detect missing or mismatched role metrics.
- Update the monitoring README and `.env.example` with the expected values and
  health-check behavior. Add focused tests for matching, missing, stale, and
  mismatched health signals.

## Constraints and implementation calls

- Keep all published ports local; monitoring stays read-only and never submits
  orders.
- The monitor-only setup remains healthy without a trade service. When the
  trade service is running, its separate role/client identity must also match.
- Implementing this requires changes in `thetagang_nt/`, monitoring scripts,
  Grafana alert rules, and focused tests, beyond the initially named
  `monitoring/run.sh` and `monitoring/README.md`.
