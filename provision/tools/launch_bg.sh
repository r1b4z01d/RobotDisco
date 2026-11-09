#!/bin/bash
# launch_bg.sh <pidfile> <logfile> <command...>
#
# Start a long-running command (a ros2 launch) detached, in its own session, with default signal dispositions, and
# print its pid. Needed from a non-interactive shell (an agent, a script, ssh with a command): bash runs `cmd &`
# there with SIGINT and SIGQUIT ignored, and some harnesses hand down SIGPIPE ignored as well. A ros2 launch that
# inherits that never reacts to `kill -INT` (seen as a 30 s "hang" with 38 processes alive, 2026-11-09), and its
# C++ children outlive a dead launch because a closed stdout pipe is no longer fatal. From an interactive terminal
# plain `ros2 launch ... &` is fine. Stop what this started with `kill -INT $(cat <pidfile>)`.
pidfile=$1; logfile=$2; shift 2
setsid -f bash -c 'echo $$ > "$0"; exec python3 -c "
import os, signal, sys
for s in (signal.SIGINT, signal.SIGPIPE, signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT):
    signal.signal(s, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])" "$@"' "$pidfile" "$@" > "$logfile" 2>&1
sleep 0.5; cat "$pidfile"
