# rd_bringup

Top-level launch files, the `config/robot.yaml` loader, supervisor units.

```python
from rd_bringup import config
config.get("arm.ip")                 # "192.168.11.21"; RD__arm__ip=... overrides for one process
```
```
rd_config hand.ports.udp             # from the shell
ros2 launch rd_bringup robot.launch.py [sim_mode:=true] [base:=true] [joystick:=true]
```

`robot.launch.py` grows a launch argument per subsystem as the phases land; `sim_mode` reaches every
include. `services/` holds the supervisor units (one per launch), installed by `provision.sh services`.
The tests check the config file's shape and that no other tracked source file types an IP address.
