"""Start and stop one supervisor unit from a node (supervisord's XML-RPC on the robot's :9999).

For work that should only run while it is needed: the SmolVLA inference server holds the model on the Orin's GPU and
is wanted for an eval or a landmark task, not at boot. Every call answers instead of raising: where there is no
supervisor (the rig, the sim) the caller hears so and carries on with whatever is already serving.
"""
from __future__ import annotations

import http.client
import socket
import time
import xmlrpc.client

from . import config

ACTIVE = ("RUNNING", "STARTING")


def default_url() -> str:
    return f"http://127.0.0.1:{config.get('robot.ports.supervisor', 9999)}/RPC2"


def _supervisor(url: str | None, timeout: float):
    class _Transport(xmlrpc.client.Transport):
        def make_connection(self, host):
            return http.client.HTTPConnection(host, timeout=timeout)
    return xmlrpc.client.ServerProxy(url or default_url(), transport=_Transport(), allow_none=True).supervisor


def state(name: str, url: str | None = None, timeout: float = 3.0) -> str | None:
    """The unit's supervisor state ("RUNNING", "STOPPED", "FATAL", ...); None when supervisor or the unit is not there."""
    try:
        return str(_supervisor(url, timeout).getProcessInfo(name)["statename"])
    except (OSError, xmlrpc.client.Error, KeyError):
        return None


def start(name: str, url: str | None = None, timeout: float = 5.0) -> tuple[bool, str]:
    """Ask supervisor to start the unit and return at once. (True, state) when it is running or starting."""
    st = state(name, url, timeout)
    if st is None:
        return False, "no supervisor or no such unit"
    if st in ACTIVE:
        return True, st
    try:
        _supervisor(url, timeout).startProcess(name, False)
    except xmlrpc.client.Fault as e:
        if "ALREADY_STARTED" not in str(e.faultString):
            return False, str(e.faultString)
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "STARTING"


def stop(name: str, url: str | None = None, timeout: float = 5.0) -> tuple[bool, str]:
    """Ask supervisor to stop the unit and return at once."""
    try:
        _supervisor(url, timeout).stopProcess(name, False)
    except xmlrpc.client.Fault as e:
        if "NOT_RUNNING" not in str(e.faultString):
            return False, str(e.faultString)
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    return True, "STOPPING"


def wait_port(host: str, port: int, timeout: float, cancelled=lambda: False) -> bool:
    """True once something accepts on host:port (the inference server listens only after its model is loaded)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cancelled():
        try:
            socket.create_connection((host, int(port)), timeout=1.0).close()
            return True
        except OSError:
            time.sleep(0.5)
    return False
