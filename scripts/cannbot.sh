#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$project_root/toolchains/node-22.23.5/bin:$PATH"
exec node "$project_root/third_party/cannbot/script/bin/cannbot.js" "$@"
