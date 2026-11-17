#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$project_root/toolchains/cannbot-venv/bin/python" \
  "$project_root/.agents/skills/knowledge-query/scripts/knowledge_query.py" \
  --knowledge-root "$project_root/third_party/cannbot-knowledge" "$@"
