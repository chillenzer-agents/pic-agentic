#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

# Start the local development Synapse created by scripts/setup_synapse.sh.
# It daemonises Synapse itself and returns as soon as the endpoint answers (or
# prints its log tail and fails), so run it in the foreground. Output always
# goes to a log file — never /dev/null, or startup errors disappear.
#
#   scripts/start_synapse.sh
#   SYNAPSE_HOME=~/synapse scripts/start_synapse.sh

set -uo pipefail

SYNAPSE_HOME="${SYNAPSE_HOME:-$HOME/synapse}"
VENV="$SYNAPSE_HOME/venv"
DATA="$SYNAPSE_HOME/data"
LOG="$SYNAPSE_HOME/synapse.log"
URL="${SYNAPSE_URL:-http://127.0.0.1:8008/_matrix/client/versions}"
WAIT_S="${SYNAPSE_WAIT_S:-30}"

if [ ! -f "$DATA/homeserver.yaml" ]; then
  echo "no config at $DATA/homeserver.yaml — run scripts/setup_synapse.sh first" >&2
  exit 1
fi

cd "$DATA" || {
  echo "cannot enter $DATA" >&2
  exit 1
}

if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
  echo "synapse already answering at $URL"
  exit 0
fi

# Detach all three streams so a leftover child can never hold a caller's pipe:
# stdout/stderr to the log, stdin from /dev/null.
nohup "$VENV/bin/python" -m synapse.app.homeserver --config-path homeserver.yaml \
  >>"$LOG" 2>&1 </dev/null &
PID=$!
echo "synapse pid $PID, log $LOG"

for _ in $(seq 1 "$WAIT_S"); do
  if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
    echo "synapse ready at $URL"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "synapse exited during startup — last log lines:" >&2
    tail -30 "$LOG" >&2
    exit 1
  fi
  sleep 1
done

echo "synapse not ready after ${WAIT_S}s — last log lines:" >&2
tail -30 "$LOG" >&2
exit 1
