#!/usr/bin/env bash
# GR00T N1.5 through LeRobot, in the LeRobot venv (/opt/vla): the extra packages its `groot` policy needs and the two
# local patches of LeRobot's groot code. Safe to run again. Used on the GPU box (training) and the robot (serving).
#
#   provision/groot_setup.sh [venv]        default /opt/vla
#
# The patches (provision/patches/lerobot-groot-*.patch, made against lerobot 0.4.4):
#   no-flash-attn      flash-attn has no wheel for torch 2.10 and does not build against it; without the package the
#                      vision-language model falls back to PyTorch's scaled-dot-product attention (same maths)
#   unequal-cameras    LeRobot stacks the camera images and needs them one size; ours are not (wrist 540x960, the
#                      others 540x540): each is padded to a square and brought to the smallest square among them
set -euo pipefail
VENV="${1:-/opt/vla}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

pkgs=("peft>=0.13,<1" "dm-tree>=0.1.8,<1" "timm>=1.0,<1.1" "ninja>=1.11.1,<2")
[ "$(uname -m)" = "x86_64" ] && pkgs+=("decord>=0.6,<1")       # the training data loader's video reader: no arm64 wheel, not needed to serve
"$VENV/bin/pip" install -q "${pkgs[@]}"

for p in "$HERE"/patches/lerobot-groot-*.patch; do
  if patch -d "$SITE" -p1 -N --dry-run -s < "$p" >/dev/null 2>&1; then
    patch -d "$SITE" -p1 -N -s -b -z .orig < "$p"
    echo "applied $(basename "$p")"
  elif patch -d "$SITE" -p1 -R --dry-run -s < "$p" >/dev/null 2>&1; then
    echo "already applied: $(basename "$p")"
  else
    echo "DOES NOT APPLY: $(basename "$p") (lerobot $("$VENV/bin/python" -c 'import lerobot; print(lerobot.__version__)'): made against 0.4.4)" >&2
    exit 1
  fi
done
"$VENV/bin/python" -c "import peft, timm, tree; from lerobot.policies.groot.modeling_groot import GrootPolicy; print('groot: ready')"
