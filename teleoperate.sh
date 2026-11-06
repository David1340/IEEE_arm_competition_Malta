#!/usr/bin/env bash
# Teleoperate every follower with its leader; the pairs and ports come from arms.json.
# Same as `python teleop.py` (see that file for the options), e.g.:
#   ./teleoperate.sh                     # each pair in its own terminal window
#   ./teleoperate.sh --here              # all in this terminal
#   ./teleoperate.sh --pair follower_blue
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$ROOT/teleop.py" "$@"
