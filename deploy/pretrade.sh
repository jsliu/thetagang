#!/bin/sh
# Preview trades and send them if you say y (thetagang-nt pretrade, in Docker):
#   ./pretrade.sh                                   # the [[strategy.trades]], sized to SPX delta 0
#   ./pretrade.sh "sell 2 AA 20261218 40P" "buy 100 MS"
#   ./pretrade.sh "buy 1 AA 20261218 40P/23P"       # a vertical: one combo order
exec docker compose -f "$(dirname "$0")/docker-compose.yml" run --rm --service-ports pretrade "$@"
