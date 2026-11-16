"""The final approach to a landmark against a simulated differential drive: from where Nav2 may leave the robot
(up to 25 cm and 14 deg off, any side) to within 3 cm and 1.5 deg, with a noisy pose and a base that lags."""
import math
import random

from rd_nav.final_approach import Params, State, step, wrap


def run(start, goal=(0.0, 0.0, 0.0), noise=0.0, lag=0.15, seconds=40.0, dt=0.05, seed=1):
    rng = random.Random(seed)
    x, y, yaw = start
    v = w = 0.0
    st, p = State(), Params()
    t, travelled = 0.0, 0.0
    while t < seconds and st.phase != "done":
        seen = (x + rng.gauss(0, noise), y + rng.gauss(0, noise), yaw + rng.gauss(0, noise * 0.5))
        cv, cw = step(st, seen, goal, p)
        assert abs(cv) <= p.v_max + 1e-9 and abs(cw) <= p.w_max + 1e-9
        a = dt / max(lag, dt)                      # the base follows its command with a lag
        v, w = v + (cv - v) * a, w + (cw - w) * a
        x, y, yaw = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, wrap(yaw + w * dt)
        travelled += abs(v) * dt
        t += dt
    return st, math.hypot(goal[0] - x, goal[1] - y), abs(wrap(goal[2] - yaw)), t, travelled


def test_reaches_the_spot_from_every_side():
    worst = (0.0, 0.0, 0.0)
    for ang in range(0, 360, 30):                  # 25 cm off in every direction, heading off by up to 14 deg
        for dyaw in (-0.25, 0.0, 0.25):
            s = (0.25 * math.cos(math.radians(ang)), 0.25 * math.sin(math.radians(ang)), dyaw)
            st, d, e, t, travelled = run(s)
            assert st.phase == "done", (ang, dyaw, st)
            assert d <= 0.035 and e <= 0.03, (ang, dyaw, d, e)
            assert travelled <= 0.45, (ang, dyaw, travelled)      # no wandering: about the offset itself
            worst = max(worst, (d, e, t))
    assert worst[2] < 30.0


def test_beside_the_goal_and_already_facing_its_way():
    """The case that stuck Nav2 on 2026-11-15: 27 cm to the side, heading right."""
    st, d, e, t, _ = run((0.06, 0.26, 0.0))
    assert st.phase == "done" and d <= 0.035 and e <= 0.03


def test_with_a_noisy_pose_and_already_there():
    for seed in range(5):
        st, d, e, _t, _ = run((0.15, -0.12, 0.2), noise=0.006, seed=seed)
        assert st.phase == "done" and d <= 0.05 and e <= 0.04, (seed, d, e)
    st, d, e, t, travelled = run((0.01, 0.01, 0.01))             # nothing to do: no motion
    assert st.phase == "done" and travelled < 0.005 and t < 1.0


def test_a_short_way_behind_is_backed_onto():
    st, d, e, _t, travelled = run((0.12, 0.0, 0.0))               # the goal is 12 cm behind the robot
    assert st.phase == "done" and d <= 0.035 and travelled <= 0.2 and st.reverse
