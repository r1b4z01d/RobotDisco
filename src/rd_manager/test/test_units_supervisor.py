"""rd_bringup.units against the rig's XML-RPC supervisor: the calls the policy runner makes for its server unit."""
import socket

from rd_bringup import units
from rd_manager.supervisor import FakeSupervisor


def test_start_stop_state_and_no_supervisor():
    sv = FakeSupervisor(units=["vla_infer_server"]).start()
    try:
        assert units.state("vla_infer_server", sv.url) == "STOPPED"
        assert units.start("vla_infer_server", sv.url) == (True, "STARTING")
        assert units.state("vla_infer_server", sv.url) == "RUNNING"
        assert units.start("vla_infer_server", sv.url) == (True, "RUNNING")          # asked twice: nothing more to do
        assert sv.calls == [("start", "vla_infer_server")]
        assert units.stop("vla_infer_server", sv.url)[0] and units.state("vla_infer_server", sv.url) == "STOPPED"
        assert units.state("no_such_unit", sv.url) is None
    finally:
        sv.stop()
    assert units.state("vla_infer_server", "http://127.0.0.1:9/RPC2", timeout=0.5) is None      # nobody there: an answer, not an exception
    assert units.start("vla_infer_server", "http://127.0.0.1:9/RPC2", timeout=0.5)[0] is False


def test_wait_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    try:
        assert units.wait_port("127.0.0.1", s.getsockname()[1], 2.0)
    finally:
        s.close()
    assert units.wait_port("127.0.0.1", 9, 0.6) is False
