#!/usr/bin/env bash
# Push this working tree to the robot and bring it up. The robot has no internet and is not git-synced:
# this rsync IS how code gets there (v1 deploy_to_rd.sh, now the whole repo).
#
#   provision/deploy_to_rd.sh                 # rsync src/ config/ provision/ ... (no build dirs), colcon build, units, restart what changed
#   provision/deploy_to_rd.sh --restart-all   # restart every unit afterwards
#   provision/deploy_to_rd.sh --dry-run       # show what would change
#
# Needs the ssh host "RD" (robot.yaml robot.ip) reachable, i.e. this machine on the robot network.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
RESTART_ALL=0; DRY=""
for a in "$@"; do
  case "$a" in
    --restart-all) RESTART_ALL=1 ;;
    --dry-run) DRY="--dry-run" ;;
    *) echo "usage: $0 [--restart-all] [--dry-run]" >&2; exit 2 ;;
  esac
done
SSH="ssh -o ConnectTimeout=8 -o BatchMode=yes"
cd "$ROOT"
$SSH RD 'echo "RD reachable: $(hostname), up $(uptime -p)"'
changed="$(rsync -ai $DRY -e "$SSH" --delete --exclude .git --exclude /rig/ --exclude build --exclude install --exclude log \
  --exclude __pycache__ --exclude '*.pyc' --exclude /src/rd_nav/landmarks/img/ --exclude '/src/rd_nav/maps/*.posegraph' \
  --exclude '/src/rd_nav/maps/*.data' --exclude /src/rd_nav/maps/active_map \
  src config provision firmware vendor README.md LICENSE RD:/home/rd/RobotDisco/ | grep -E '^[<>c]' || true)"
echo "$changed" | sed 's/^/  /' | head -40
[ -n "$DRY" ] && exit 0
pkgs="$(echo "$changed" | grep -oE 'src/rd_[a-z_]+' | sort -u | sed 's#src/##' | tr '\n' ' ')"
units="$(echo "$changed" | grep -oE 'src/rd_[a-z_]+' | sort -u | sed 's#src/##' | grep -vE 'rd_interfaces|rd_description|rd_bringup' | tr '\n' ' ' || true)"
echo "packages: ${pkgs:-none}"
remote="set -e; cd ~/RobotDisco; source /opt/ros/jazzy/setup.bash
echo '=== build'; colcon build --symlink-install ${pkgs:+--packages-up-to $pkgs} 2>&1 | tail -3
echo '=== units'; sudo -n cp src/rd_bringup/services/*.conf /etc/supervisor/conf.d/ && sudo -n supervisorctl reread >/dev/null && sudo -n supervisorctl update >/dev/null || true
if [ $RESTART_ALL = 1 ]; then sudo -n supervisorctl restart all | tail -20; fi
for u in $units; do [ $RESTART_ALL = 1 ] || { sudo -n supervisorctl status \$u >/dev/null 2>&1 && sudo -n supervisorctl restart \$u | tail -1 || true; }; done
sleep 5; sudo -n supervisorctl status | awk '{print \$1, \$2}'"
$SSH RD "$remote"
