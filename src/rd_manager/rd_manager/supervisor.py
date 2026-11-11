"""supervisord over XML-RPC (the Services tab, the Maps tab's unit restarts, the activity log). v1 webui
server.py's supervisor section, ROS-free. The URL comes from robot.yaml (robot.ip : robot.ports.supervisor)."""
from __future__ import annotations

import http.client
import re
import time
import xmlrpc.client
from concurrent.futures import ThreadPoolExecutor

_XML_FORBIDDEN = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_XML_RESTORE = {0xE000 + i: chr(i) for i in range(0x20) if i not in (9, 10, 13)}

# the ordered restart of the base stack, with settle times (v1: a unit restarted before the base settles
# ends up bound to publishers that no longer exist)
BASE_STACK = [("rd_base", 8.0), ("rd_perception", 4.0), ("rd_nav", 6.0)]
ACTIVITY_UNITS = [("rd_nav", "nav"), ("rd_base", "base"), ("rd_perception", "sense"), ("rd_arm", "arm"), ("rd_vla", "vla"),
                  ("rd_manager", "mgr"), ("rd_voice", "voice"), ("rd_face", "face"), ("rd_audio", "say"), ("vla_infer_server", "infer")]
_ROS_LINE = re.compile(r"^\[(INFO|WARN|WARNING|ERROR|FATAL|DEBUG)\]\s*\[(\d{9,}(?:\.\d+)?)\]\s*\[([^\]]+)\]:\s*(.*)$")
_ROS_LINE2 = re.compile(r"^\[([\w.\-]+)-\d+\]\s*\[(INFO|WARN|WARNING|ERROR|FATAL|DEBUG)\]\s*\[(\d{9,}(?:\.\d+)?)\]\s*\[([^\]]+)\]:\s*(.*)$")
_PY_LEVEL = re.compile(r"\b(ERROR|CRITICAL|Traceback|WARNING|WARN)\b")
_ACT_NOISE = re.compile(r"(Publishing:|Received|get_state|lifecycle_manager.*(Waiting for service|Creating bond)|"
                        r"Transform data too old|Failed to bring up all nodes|GET /|POST /|HTTP/1\.[01]\" 200|"
                        r"Message Filter dropping|UDP|keepalive|heartbeat|FIFO RT|"
                        r"Velocity command timed out)", re.I)     # diff_cont's once-a-second "nobody is driving": not news
_ACT_WAIT = re.compile(r"(wait|stop|abort|fail|timeout|time out|stuck|reject|cancel|no path|collision|blocked|"
                       r"recover|retry|lost|not reachable|unavailable)", re.I)


def _xml_restore(v):
    if isinstance(v, str):
        return v.translate(_XML_RESTORE)
    if isinstance(v, list):
        return [_xml_restore(x) for x in v]
    if isinstance(v, dict):
        return {k: _xml_restore(x) for k, x in v.items()}
    return v


class Supervisor:
    def __init__(self, url: str, timeout: float = 4.0):
        self.url = url
        self.timeout = timeout
        self.settle_scale = 1.0                 # the rig's tests set 0: no waiting between the base units
        self.job = {"running": False, "log": [], "t": 0.0}

    def proxy(self):
        timeout = self.timeout

        class _Transport(xmlrpc.client.Transport):
            def make_connection(self, host):
                return http.client.HTTPConnection(host, timeout=timeout)

            def getparser(self):
                parser, unmarshaller = super().getparser()
                feed, close = parser.feed, unmarshaller.close
                parser.feed = lambda data: feed(_XML_FORBIDDEN.sub(lambda m: b"&#x%04X;" % (0xE000 + m.group()[0]), data))
                unmarshaller.close = lambda: _xml_restore(close())
                return parser, unmarshaller
        return xmlrpc.client.ServerProxy(self.url, transport=_Transport(), allow_none=True).supervisor

    def stop_wait(self, sv, name: str, timeout: float = 25.0) -> str:
        try:
            sv.stopProcess(name, False)
        except Exception:  # noqa: BLE001
            pass
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = sv.getProcessInfo(name)["statename"]
            except Exception:  # noqa: BLE001
                st = "?"
            if st in ("STOPPED", "EXITED", "FATAL", "UNKNOWN"):
                return st
            time.sleep(0.5)
        return "STOPPING"

    def services(self) -> dict:
        try:
            infos = self.proxy().getAllProcessInfo()
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "error": f"{type(e).__name__}: {e}", "url": self.url, "job": self.job}
        now = time.time()
        out = []
        for i in infos:
            running = i["statename"] == "RUNNING"
            out.append({"name": i["name"], "group": i["group"], "state": i["statename"], "pid": i["pid"] or None,
                        "uptime_s": int(now - i["start"]) if running and i["start"] else None,
                        "description": i["description"], "stdout": i.get("stdout_logfile"), "stderr": i.get("stderr_logfile")})
        return {"reachable": True, "url": self.url, "services": out, "job": self.job}

    def states(self, names=None) -> dict:
        try:
            return {p["name"]: p["statename"] for p in self.proxy().getAllProcessInfo() if names is None or p["name"] in names}
        except Exception as e:  # noqa: BLE001
            return {"error": f"supervisor: {e}"}

    def start_wait(self, sv, name: str, timeout: float = 90.0) -> str:
        """Start a unit and wait until supervisor calls it RUNNING. Asked to wait itself (startProcess(name, True)),
        supervisor answers only after the unit's startsecs, 15 s for the launches, and this client gives a call 4 s:
        every map switch on the robot ended 'failed: timed out' with the unit up and fine (2026-11-11)."""
        try:
            sv.startProcess(name, False)
        except xmlrpc.client.Fault as e:
            if "ALREADY_STARTED" not in str(e.faultString):
                raise
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st = sv.getProcessInfo(name)["statename"]
            except Exception:  # noqa: BLE001
                st = "?"
            if st in ("RUNNING", "FATAL", "EXITED", "BACKOFF"):
                if st != "RUNNING":
                    raise RuntimeError(f"{name} did not start: {st}")
                return st
            time.sleep(0.5 * self.settle_scale if self.settle_scale else 0.01)
        raise RuntimeError(f"{name} still starting after {timeout:.0f} s")

    def action(self, name: str, action: str) -> dict:
        sv = self.proxy()
        if action == "start":
            self.start_wait(sv, name)
        elif action == "stop":
            self.stop_wait(sv, name)
        elif action == "restart":
            self.stop_wait(sv, name)
            self.start_wait(sv, name)
        else:
            raise ValueError("action must be start|stop|restart")
        return {"ok": True, "name": name, "action": action}

    def log(self, name: str, stream: str = "stdout", lines: int = 80) -> dict:
        sv = self.proxy()
        fn = sv.tailProcessStderrLog if stream == "stderr" else sv.tailProcessStdoutLog
        err = None
        for length in (60000, 59999, 59998, 59997):
            try:
                data, offset, overflow = fn(name, 0, length)
                break
            except xmlrpc.client.ProtocolError as e:
                err = e
        else:
            raise RuntimeError(f"supervisor: {err}")
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", data)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return {"name": name, "stream": stream, "lines": text.splitlines()[-max(1, min(lines, 2000)):]}

    def restart_base(self) -> dict:
        import threading
        if self.job["running"]:
            raise RuntimeError("already running")

        def run():
            self.job.update(running=True, log=[], t=time.time())
            try:
                sv = self.proxy()
                for name, settle in BASE_STACK:
                    self.job["log"].append(f"restarting {name}")
                    self.stop_wait(sv, name)
                    self.start_wait(sv, name)
                    if settle * self.settle_scale > 0:
                        self.job["log"].append(f"settling {settle:.0f} s")
                        time.sleep(settle * self.settle_scale)
                self.job["log"].append("done; set the initial pose again if the robot was localized")
            except Exception as e:  # noqa: BLE001
                self.job["log"].append(f"failed: {e}")
            finally:
                self.job["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True}

    # ---- the activity log
    def _unit_tail(self, sv, name, stream, nbytes=16000) -> list:
        fn = sv.tailProcessStderrLog if stream == "stderr" else sv.tailProcessStdoutLog
        for length in (nbytes, nbytes - 1, nbytes - 2, nbytes - 3):
            try:
                data = fn(name, 0, length)[0]
                break
            except Exception:  # noqa: BLE001
                data = None
        if not data:
            return []
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", data)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text.splitlines()[1:]

    @staticmethod
    def activity_lines(short: str, lines: list) -> list:
        out, last_t = [], 0.0
        for ln in lines:
            ln = ln.rstrip()
            if not ln.strip():
                continue
            m = _ROS_LINE.match(ln)
            m2 = None if m else _ROS_LINE2.match(ln)
            if m:
                lvl, t, node, msg = m.group(1), float(m.group(2)), m.group(3), m.group(4)
            elif m2:
                lvl, t, node, msg = m2.group(2), float(m2.group(3)), m2.group(4), m2.group(5)
            else:
                lvl, t, node, msg = ("ERROR" if _PY_LEVEL.search(ln) and "WARN" not in ln.upper() else
                                     "WARN" if "WARN" in ln.upper() else "INFO"), last_t, "", ln
            if lvl == "DEBUG" or _ACT_NOISE.search(msg):
                continue
            last_t = t or last_t
            out.append({"t": t or last_t, "unit": short, "node": node.split(".")[-1], "level": "WARN" if lvl == "WARNING" else lvl,
                        "msg": msg[:400], "wait": bool(_ACT_WAIT.search(msg))})
        return out

    def activity(self, lines: int = 200, units=None) -> dict:
        try:
            infos = {p["name"]: p for p in self.proxy().getAllProcessInfo()}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "error": f"{type(e).__name__}: {e}", "lines": [], "units": []}
        now = time.time()
        chosen = [(n, s) for n, s in ACTIVITY_UNITS if n in infos and (units is None or s in units)]

        def pull(item):
            n, s = item
            if infos[n]["statename"] not in ("RUNNING", "STARTING", "BACKOFF", "FATAL", "EXITED"):
                return []
            sv = self.proxy()
            rows = []
            for stream in ("stdout", "stderr"):
                try:
                    rows += self.activity_lines(s, self._unit_tail(sv, n, stream))
                except Exception:  # noqa: BLE001
                    pass
            return rows
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = [r for part in ex.map(pull, chosen) for r in part]
        rows.sort(key=lambda r: r["t"])
        merged: list = []
        for r in rows:
            p = merged[-1] if merged else None
            if p and p["unit"] == r["unit"] and p["msg"] == r["msg"] and p["level"] == r["level"]:
                p["count"] = p.get("count", 1) + 1
                p["t"] = r["t"]
            else:
                merged.append(dict(r))
        return {"reachable": True, "now": now, "lines": merged[-max(10, min(lines, 800)):],
                "units": [{"unit": s, "name": n, "state": infos[n]["statename"],
                           "uptime_s": int(now - infos[n]["start"]) if infos[n]["statename"] == "RUNNING" and infos[n]["start"] > 1e9 else None}
                          for n, s in chosen]}


class FakeSupervisor:
    """An XML-RPC supervisord for the rig: the same calls over a real socket, state in memory."""

    def __init__(self, units=None, port: int = 0):
        from xmlrpc.server import SimpleXMLRPCServer
        import threading
        self.units = {u: {"name": u, "group": u, "statename": "STOPPED", "pid": 0, "start": 0, "description": "", "stdout_logfile": f"/tmp/{u}.log",
                          "stderr_logfile": f"/tmp/{u}.err"} for u in (units or [n for n, _ in ACTIVITY_UNITS])}
        self.logs = {u: ["[INFO] [1700000000.000] [node]: started"] for u in self.units}
        self.calls = []
        self.server = SimpleXMLRPCServer(("127.0.0.1", port), allow_none=True, logRequests=False)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/RPC2"

        class API:
            def getAllProcessInfo(api_self):
                return list(self.units.values())

            def getProcessInfo(api_self, name):
                return self.units[name]

            def startProcess(api_self, name, wait=True):
                self.calls.append(("start", name))
                self.units[name].update(statename="RUNNING", pid=4242, start=int(time.time()))
                return True

            def stopProcess(api_self, name, wait=True):
                self.calls.append(("stop", name))
                self.units[name].update(statename="STOPPED", pid=0)
                return True

            def tailProcessStdoutLog(api_self, name, offset, length):
                return ["\n".join(self.logs[name]), 0, False]

            def tailProcessStderrLog(api_self, name, offset, length):
                return ["", 0, False]
        self.server.register_instance(API(), allow_dotted_names=False)
        self.server.register_multicall_functions()
        # supervisor namespaces its API as "supervisor.<method>"
        inst = API()
        for n in ("getAllProcessInfo", "getProcessInfo", "startProcess", "stopProcess", "tailProcessStdoutLog", "tailProcessStderrLog"):
            self.server.register_function(getattr(inst, n), "supervisor." + n)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
