#!/bin/sh
set -eu

CONFIG_FILE="${FRONTEND_DIST_DIR:-/app/frontend/dist}/config.js"
API_URL="${BACKEND_HOST:-}"

cat > "$CONFIG_FILE" <<EOF
window.__LLM_COUNCIL_CONFIG__ = {
  apiUrl: "$API_URL",
};
EOF

exec "$@"
