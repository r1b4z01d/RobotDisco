"""rd_manager: the web console.

    manager --port 8092                       # the robot: ROS bridge on, supervisor on the robot's :9999
    manager --port 8081 --no-ros              # the GPU box: datasets / models / evals / training only, robot tabs
                                              # pointed at the robot's instance (--robot-url)
    manager --sim true                        # the rig: ROS bridge on, no supervisor expected
"""
from __future__ import annotations
from rd_bringup.process import exit_with_parent

import argparse
import os
import sys


def main(argv=None) -> int:
    exit_with_parent()   # end with the launch that started us (rd_bringup.process)
    from rd_bringup import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(config.get("robot.ports.manager", 8092)))
    ap.add_argument("--no-ros", action="store_true", help="no ROS bridge (the GPU box)")
    ap.add_argument("--sim", default="false", help="true in the rig: no supervisor, the robot tabs point here")
    ap.add_argument("--robot-url", default=f"http://{config.get('robot.ip')}:{config.get('robot.ports.manager', 8092)}",
                    help="where the robot's manager runs (the GPU box's Robot/Live tabs)")
    ap.add_argument("--supervisor-url", default=f"http://{config.get('robot.ip')}:{config.get('robot.ports.supervisor', 9999)}/RPC2")
    panels_ip, panels_port = str(config.get("panels.ip", "")), int(config.get("panels.ports.http", 80))
    ap.add_argument("--panels-url", default=(f"http://{panels_ip}" + ("" if panels_port == 80 else f":{panels_port}") + "/") if panels_ip else "",
                    help="the WLED controller behind the LED matrix panels (the Panels tab); empty = no panels")
    ap.add_argument("--data", default=str(config.data_dir()),
                    help="datasets, train runs and evals live under <data>/outputs")
    ap.add_argument("--maps-dir", default=os.environ.get("RD_MAPS_DIR", ""))
    ap.add_argument("--landmarks-dir", default=os.environ.get("RD_LANDMARKS_DIR", ""))
    ap.add_argument("--routes-dir", default=os.environ.get("RD_ROUTES_DIR", ""))
    args, _ros_args = ap.parse_known_args(argv)      # launch_ros appends --ros-args -r __node:=...; the bridge names its own node
    sim = str(args.sim).lower() in ("1", "true", "yes")
    from .server import build_app
    app = build_app(ros=not args.no_ros, sim=sim, data_dir=args.data, robot_url=args.robot_url,
                    supervisor_url="" if sim else args.supervisor_url, maps_dir=args.maps_dir or None,
                    landmarks_dir=args.landmarks_dir or None, routes_dir=args.routes_dir or None, panels_url=args.panels_url)
    import uvicorn
    print(f"rd_manager: http://{args.host}:{args.port}  (ros {'on' if not args.no_ros else 'off'}, {'sim' if sim else 'robot'}, data {args.data})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
