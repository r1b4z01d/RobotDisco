# rd_bringup

**Not started** (phase 1). Top-level launch files (`robot.launch.py`: base + perception + nav + manager;
`arm.launch.py`; `face.launch.py`), the supervisor units (one per launch), udev and netplan install, and
the loader that exports `config/robot.yaml` to every launch. Replaces v1's `services/*.conf` and the
robot's hand-written `run.sh`.
