#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

# Create a throwaway local Synapse for development (the homeserver the
# level-2 smoke test talks to). Idempotent: re-running reuses the venv and
# config and only appends the development settings when they are missing.
#
#   scripts/setup_synapse.sh
#   SYNAPSE_HOME=~/synapse scripts/setup_synapse.sh
#
# Then start it with scripts/start_synapse.sh.

set -euo pipefail

SYNAPSE_HOME="${SYNAPSE_HOME:-$HOME/synapse}"
VENV="$SYNAPSE_HOME/venv"
DATA="$SYNAPSE_HOME/data"
SERVER_NAME="${SYNAPSE_SERVER_NAME:-localhost}"
SYNAPSE_VERSION="${SYNAPSE_VERSION:-1.161.0}"

mkdir -p "$SYNAPSE_HOME"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found — install it (https://docs.astral.sh/uv/) or create $VENV yourself" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv at $VENV ..."
  uv venv --python 3.13 "$VENV"
fi

echo "installing matrix-synapse==$SYNAPSE_VERSION ..."
uv pip install --python "$VENV/bin/python" "matrix-synapse==$SYNAPSE_VERSION"

if [ ! -f "$DATA/homeserver.yaml" ]; then
  echo "generating config in $DATA ..."
  mkdir -p "$DATA"
  (
    cd "$DATA"
    "$VENV/bin/python" -m synapse.app.homeserver \
      --server-name "$SERVER_NAME" \
      --config-path homeserver.yaml \
      --generate-config --report-stats=no
  )
fi

# Development settings the generated config omits. Guarded so re-runs are safe.
if grep -q "^enable_registration:" "$DATA/homeserver.yaml"; then
  echo "development settings already present; leaving config unchanged."
else
  echo "appending development settings to $DATA/homeserver.yaml ..."
  cat >>"$DATA/homeserver.yaml" <<'YAML'

# --- development settings (added by scripts/setup_synapse.sh) ---
enable_registration: true
enable_registration_without_verification: true
suppress_key_server_warning: true

# Disable rate limiting: repeated test registrations otherwise fail with
# M_LIMIT_EXCEEDED (retry_after_ms ~300000).
rc_registration:
  per_second: 10000
  burst_count: 10000
rc_login:
  address:
    per_second: 10000
    burst_count: 10000
  account:
    per_second: 10000
    burst_count: 10000
  failed_attempts:
    per_second: 10000
    burst_count: 10000
rc_joins:
  local:
    per_second: 10000
    burst_count: 10000
rc_invites:
  per_room:
    per_second: 10000
    burst_count: 10000
rc_message:
  per_second: 10000
  burst_count: 10000
YAML
fi

echo
echo "done. Start it with:"
echo "  SYNAPSE_HOME=$SYNAPSE_HOME scripts/start_synapse.sh"
