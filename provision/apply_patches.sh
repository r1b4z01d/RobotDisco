#!/usr/bin/env bash
# Apply provision/patches/<submodule>-<topic>.patch to vendor/<submodule>. A vendored package is a submodule
# pinned to an upstream commit, so every fresh checkout has to re-apply our patches: provision.sh (workspace),
# gpu/provision_gpu.sh (workspace) and rig/rig.sh (build) all call this before colcon. Idempotent: a patch
# already in the tree is skipped; one that no longer applies is an error (the pin moved: rebase the patch).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
status=0
for p in "$ROOT"/provision/patches/*.patch; do
  [ -f "$p" ] || continue
  name="${p##*/}"; sub="${name%%-*}"; dir="$ROOT/vendor/$sub"
  if [ ! -e "$dir/.git" ]; then
    echo "patch $name: vendor/$sub is not checked out (git submodule update --init)" >&2; status=1; continue
  fi
  if git -C "$dir" apply --check --reverse "$p" >/dev/null 2>&1; then
    echo "vendor/$sub: $name already applied"
  elif git -C "$dir" apply "$p"; then
    echo "vendor/$sub: applied $name"
  else
    echo "vendor/$sub: $name did not apply" >&2; status=1
  fi
done
exit $status
