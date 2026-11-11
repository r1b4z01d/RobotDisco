"""The RoboClaw hardware interface against a board that goes quiet (the retry patch in provision/patches).

A fake RoboClaw answers packet serial on a pty: READ_M1_M2_ENC (78) and DRIVE_M1_M2_SGN_SPD (37), CRC16 as the
board computes it. The real controller manager runs the real plugin on the pty's other end. The fake then
- drops one request: the second attempt absorbs it, nobody notices;
- stays silent for a quarter of a second: tolerated, the hardware stays active, the log says recovered;
- stays silent for good: the hardware reports its error within about a second.
Before the patch the first dropped request deactivated the hardware and both controllers until the launch was
restarted (the robot, 2026-11-10).
"""
import os
import select
import struct
import subprocess
import tempfile
import termios
import threading
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_prefix, get_package_share_directory
from controller_manager_msgs.srv import ListHardwareComponents
from sensor_msgs.msg import JointState

DOMAIN = "87"       # its own ROS domain: the controller manager's default name must not meet another stack's


def crc16(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class FakeRoboClaw:
    """Packet serial address 128 on the master end of a pty. `drop` requests are swallowed; `silent_until` swallows all."""

    def __init__(self):
        self.master, slave = os.openpty()
        attrs = termios.tcgetattr(slave)
        attrs[0] = attrs[1] = attrs[3] = 0                      # raw both ways before anyone opens it
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        self.port = os.ttyname(slave)
        self._slave = slave
        self.drop, self.silent_until, self.ticks, self.requests, self.speeds = 0, 0.0, 0, 0, (0, 0)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        buf = b""
        while not self._stop.is_set():
            if not select.select([self.master], [], [], 0.05)[0]:
                continue
            try:
                buf += os.read(self.master, 256)
            except OSError:
                return
            while len(buf) >= 2:
                if buf[1] == 78:
                    need, reply = 2, None
                elif buf[1] == 37:
                    need, reply = 12, b"\xff"
                else:
                    buf = buf[1:]                               # not a request we know: resynchronise
                    continue
                if len(buf) < need:
                    break
                req, buf = buf[:need], buf[need:]
                self.requests += 1
                if self.drop > 0 or time.monotonic() < self.silent_until:
                    self.drop = max(0, self.drop - 1)
                    continue
                if reply is None:
                    self.ticks += 100
                    data = struct.pack(">ii", self.ticks, -self.ticks)
                    reply = data + struct.pack(">H", crc16(req + data))
                else:
                    self.speeds = struct.unpack(">ii", req[2:10])
                os.write(self.master, reply)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=1.0)
        os.close(self.master)
        os.close(self._slave)


@pytest.fixture(scope="module")
def rig():
    try:
        get_package_share_directory("roboclaw_hardware_interface")
    except PackageNotFoundError:
        pytest.skip("roboclaw_hardware_interface is not built here")
    fake = FakeRoboClaw()
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    urdf = subprocess.run(["xacro", os.path.join(get_package_share_directory("rd_description"), "urdf", "robot.urdf.xacro"),
                           "sim_mode:=false", f"roboclaw_port:={fake.port}", "roboclaw_baud:=115200"],
                          capture_output=True, text=True, check=True).stdout
    tmp = tempfile.TemporaryDirectory()
    params = Path(tmp.name) / "description.yaml"
    # Jazzy's controller manager takes the URDF from the robot_description topic: the state publisher latches it there
    params.write_text(yaml.safe_dump({"controller_manager": {"ros__parameters": {"robot_description": urdf}},
                                      "robot_state_publisher": {"ros__parameters": {"robot_description": urdf}}}))
    # the executables themselves, not `ros2 run`: its wrapper does not pass a terminate on to the node
    exe = lambda pkg, name: os.path.join(get_package_prefix(pkg), "lib", pkg, name)  # noqa: E731
    rsp = subprocess.Popen([exe("robot_state_publisher", "robot_state_publisher"), "--ros-args", "--params-file", str(params)],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(Path(tmp.name) / "cm.log", "w+")
    controllers = os.path.join(get_package_share_directory("rd_base"), "config", "controllers.yaml")
    cm = subprocess.Popen([exe("controller_manager", "ros2_control_node"), "--ros-args",
                           "--params-file", str(params), "--params-file", controllers], env=env, stdout=log, stderr=subprocess.STDOUT)
    spawner = subprocess.run(["ros2", "run", "controller_manager", "spawner", "joint_broad", "--controller-manager-timeout", "30"],
                             env=env, capture_output=True, text=True, timeout=60)
    os.environ["ROS_DOMAIN_ID"] = DOMAIN
    rclpy.init()
    node = rclpy.create_node("roboclaw_link_test")
    states = []
    node.create_subscription(JointState, "/joint_states", states.append, 10)
    client = node.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components")

    def hardware_state() -> str:
        assert client.wait_for_service(timeout_sec=10.0)
        f = client.call_async(ListHardwareComponents.Request())
        rclpy.spin_until_future_complete(node, f, timeout_sec=5.0)
        return f.result().component[0].state.label if f.result() and f.result().component else "?"

    def spin(seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def text() -> str:
        log.flush()
        return Path(log.name).read_text()

    yield type("Rig", (), {"fake": fake, "states": states, "hardware_state": staticmethod(hardware_state), "spin": staticmethod(spin),
                           "log": staticmethod(text), "spawner": spawner})
    node.destroy_node()
    rclpy.try_shutdown()
    for proc in (cm, rsp):
        proc.terminate()
    for proc in (cm, rsp):
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    fake.close()
    log.close()
    tmp.cleanup()


def test_1_reads_the_board(rig):
    assert rig.spawner.returncode == 0, rig.spawner.stdout + rig.spawner.stderr + rig.log()
    assert rig.hardware_state() == "active"
    rig.spin(1.0)
    wheel = [s.position[s.name.index("left_wheel_joint")] for s in rig.states if "left_wheel_joint" in s.name]
    assert len(wheel) > 10 and wheel[-1] > wheel[0], "the position state does not follow the board's encoder"
    assert "RoboClaw read" not in rig.log() and "RoboClaw write" not in rig.log()      # no failed exchange so far


def test_2_one_lost_reply_is_absorbed_by_the_second_attempt(rig):
    for _ in range(5):
        rig.fake.drop = 1
        rig.spin(0.3)
    assert rig.hardware_state() == "active"
    assert "tolerating" not in rig.log(), rig.log()[-800:]


def test_3_a_quarter_second_of_silence_is_tolerated(rig):
    before = len(rig.states)
    rig.fake.silent_until = time.monotonic() + 0.25
    rig.spin(1.5)
    assert rig.hardware_state() == "active"
    log = rig.log()
    assert "tolerating failures" in log and "recovered after" in log, log[-800:]
    assert len(rig.states) > before + 10, "joint states stopped"


def test_4_a_dead_link_is_reported_within_about_a_second(rig):
    rig.fake.silent_until = time.monotonic() + 3600
    t0 = time.monotonic()
    state = "active"
    while time.monotonic() - t0 < 5.0 and state == "active":
        rig.spin(0.2)
        state = rig.hardware_state()
    assert state != "active", "a dead RoboClaw link was never reported"
    assert time.monotonic() - t0 < 2.5
    assert "giving up" in rig.log()
