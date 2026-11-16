# IB Connection Health Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report thetagang healthy only when its required IB process roles are connected to the expected account and client IDs.

**Architecture:** Each thetagang-nt process publishes a low-cardinality IB session health metric with role, effective client ID, observed account, and connection state. The monitoring script compares current metrics with expected account/base client ID values, while Grafana periodically alerts when an active role is missing or unhealthy.

**Tech Stack:** Python 3.12+, NautilusTrader, Prometheus client, Bash, Grafana alert provisioning JSON, pytest.

**Spec:** [2026-11-16-ib-health-check-design.md](../specs/2026-11-16-ib-health-check-design.md)

## Global Constraints

- Monitoring remains read-only and never submits orders.
- Published service ports remain bound to `127.0.0.1`.
- The monitor-only setup remains healthy without a trade service.
- When the trade service is running, its separate role/client identity must also match.
- Do not add a direct IB connection from the monitoring script.
- Do not change trading behavior based on monitoring health.

## Review Focus

- A live IB connection to an account other than the expected account must not report healthy; test mismatch.
- A missing or stale scrape must not report healthy; test absence and staleness.
- Monitor-only startup must not be reported as a failed trade session; test the optional trade role.
- A role/client-ID mismatch must fail the one-shot health check; test both IDs and role labels.
- Disconnect followed by reconnect must update health rather than leave a stale success; test both state transitions.

---

### Task 1: Publish per-role IB session health

**Files:**
- Modify: `thetagang_nt/node.py`
- Modify: `thetagang_nt/monitor.py`
- Modify: `thetagang_nt/strategy.py`
- Test: `tests_nt/test_connection_health.py`

**Interfaces:**
- Produces Prometheus gauge `thetagang_ib_session_healthy{role,client_id,account}`. Its value is `1` only while the process's own IB session is connected and its observed account matches its configured account; otherwise it is `0`.
- `role` is the effective node role (`monitor` or `trade`), `client_id` is the effective client ID from `split()`, and `account` is the account observed from the active IB session.
- Publish the gauge on the existing role-specific metrics endpoint. Keep the existing `thetagang_last_update_timestamp_seconds` metric unchanged.

- [ ] **Step 1: Add failing tests for health transitions and identity**

Test that the metric starts unhealthy, becomes healthy only for a connected session on the configured account, returns unhealthy on disconnect, and carries the effective role/client ID/account labels. Include an account mismatch case.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run pytest -q tests_nt/test_connection_health.py`
Expected: FAIL because the health metric and state updates do not exist yet.

- [ ] **Step 3: Implement the session-health gauge**

Use the NautilusTrader IB connection lifecycle and account state for the process itself, not the read-only side connection in `AccountExtras`. Derive the role and client ID from the same values passed to `node_config()`. Do not add a dependency.

- [ ] **Step 4: Run the focused tests**

Run: `uv run pytest -q tests_nt/test_connection_health.py`
Expected: PASS for connected, disconnected, wrong-account, and role/client identity cases.

### Task 2: Add expected-identity checks to monitoring

**Files:**
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `run.sh`
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `.env.example`
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `test_config.py`

**Interfaces:**
- Add `THETAGANG_EXPECTED_IB_ACCOUNT_ID` and `THETAGANG_EXPECTED_IB_CLIENT_ID` to the monitoring environment example.
- Add `./run.sh health`: query Prometheus's current `thetagang` and `thetagang-strategy` series; require the monitor role at base client ID + 1 and the trade role at the base client ID when that target is running.
- Return nonzero for an active target with a missing, stale, unhealthy, wrong-account, wrong-role, or wrong-client-ID metric. If the trade target is not running, report it as not expected rather than unhealthy.
- After `./run.sh up`, run a startup check for active bot targets; do not prevent the monitoring stack from starting when thetagang itself is not running.

- [ ] **Step 1: Add failing tests for health query and outcomes**

Test a matching monitor-only setup, a matching monitor-plus-trade setup, wrong account/client ID/role, missing metrics, stale timestamps, and an absent optional trade target. Stub the Prometheus API response; do not require Docker or live IB.

- [ ] **Step 2: Run the monitoring tests and verify they fail**

Run from the monitoring worktree: `uv run --project ../thetagang/.worktrees/ib-health-check pytest -q test_config.py`
Expected: FAIL because `health` and the expected identity settings are not implemented.

- [ ] **Step 3: Implement `run.sh health` and startup invocation**

Query the local Prometheus HTTP API with bounded timeouts. Quote account values safely in the query and preserve the existing script's error reporting style.

- [ ] **Step 4: Run the monitoring tests**

Run from the monitoring worktree: `uv run --project ../thetagang/.worktrees/ib-health-check pytest -q test_config.py`
Expected: PASS for every configured health state and startup behavior.

### Task 3: Add periodic alerts and operator instructions

**Files:**
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `grafana/provisioning/alerting/rules.yaml`
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `test_config.py`
- Modify in separate `/Users/zhenliu/Documents/project/monitoring` repository: `README.md`

**Interfaces:**
- Add a thetagang alert that fires when a role target has been up recently but its current session-health metric is absent, disconnected, or reports an account/role mismatch.
- Reuse the existing `thetagang` alert window and live-account label. Preserve the existing no-data behavior for projects that were not running.
- Document expected environment values, `./run.sh health`, startup behavior, periodic alerting, and the optional monitor-only mode.

- [ ] **Step 1: Add failing config tests for the periodic health alert**

Assert the rule checks session health for each active role, keeps the existing schedule/account labels and no-data behavior, and does not alert when the project was not running.

- [ ] **Step 2: Run monitoring tests and verify they fail**

Run from the monitoring worktree: `uv run --project ../thetagang/.worktrees/ib-health-check pytest -q test_config.py`
Expected: FAIL because the health alert and documentation are not present.

- [ ] **Step 3: Add the alert rule and README instructions**

Use the existing JSON-formatted Grafana provisioning pattern and Telegram message template. Keep the rule independent of bot downtime history so an intentionally stopped project remains quiet.

- [ ] **Step 4: Run focused repository tests**

Run: `uv run pytest -q tests_nt/test_connection_health.py`
Run from the monitoring worktree: `uv run --project ../thetagang/.worktrees/ib-health-check pytest -q test_config.py`
Expected: PASS. If NautilusTrader is unavailable in the local environment, report the skipped runtime tests rather than installing dependencies.
