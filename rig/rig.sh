#!/usr/bin/env bash
# The off-robot test rig. Runs the workspace in a ROS 2 Jazzy container (Docker; Colima on the Mac).
#
#   rig/rig.sh image            build (or rebuild) the image from rig/Dockerfile
#   rig/rig.sh build [pkgs]     colcon build --symlink-install (all packages, or the named ones)
#   rig/rig.sh test  [pkgs]     colcon test + result summary
#   rig/rig.sh pytest <path>    run pytest on one package's tests, verbose
#   rig/rig.sh sh               interactive shell with the workspace sourced
#   rig/rig.sh run <cmd ...>    any command inside the sourced container
#
# build/ install/ log/ live in rig/.colcon/ on the host (bind-mounted), so rebuilds are incremental and
# the source tree stays clean. IMAGE=... overrides the image tag.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-robotdisco-rig:jazzy}"
mkdir -p "$HERE/.colcon"/{build,install,log}

run() {
  docker run --rm -i $([ -t 0 ] && echo -t) --ulimit core=0 \
    -v "$ROOT:/ws" \
    -v "$HERE/.colcon/build:/ws/build" -v "$HERE/.colcon/install:/ws/install" -v "$HERE/.colcon/log:/ws/log" \
    -e RD_ROBOT_CONFIG=/ws/config/robot.yaml \
    -w /ws "$IMAGE" "$*"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  image)  docker build -t "$IMAGE" -f "$HERE/Dockerfile" "$ROOT" ;;
  build)  "$ROOT/provision/apply_patches.sh"
          run "colcon build --symlink-install --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release ${*:+--packages-up-to $*}" ;;
  test)   run "colcon test --executor sequential --event-handlers console_cohesion+ --pytest-args -x --timeout=900 ${*:+--packages-select $*}; colcon test-result --verbose | tail -40" ;;
  pytest) run "python3 -m pytest -x -v --timeout=120 $*" ;;
  sh)     run "bash" ;;
  run)    run "$*" ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
