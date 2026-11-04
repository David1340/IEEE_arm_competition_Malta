#!/usr/bin/env bash
# Open two terminals and teleoperate both arms:
#   follower_pink  <- leader_blue
#   follower_blue  <- leader_green
# Ports are read from ports/<name>.txt (edit those files if the USB ports change).
#
# Usage:
#   ./teleoperate.sh               # uses the conda env "lerobot"
#   CONDA_ENV=myenv ./teleoperate.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-lerobot}"

port() {
    local f="$ROOT/ports/$1.txt"
    [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
    tr -d '[:space:]' < "$f"
}

teleop_cmd() {  # follower_id leader_id
    echo "lerobot-teleoperate \
--robot.type=so101_follower --robot.port=$(port "$1") --robot.id=$1 \
--teleop.type=so101_leader --teleop.port=$(port "$2") --teleop.id=$2"
}

open_terminal() {  # title command
    local title="$1" cmd="$2"
    # -i loads ~/.bashrc so conda is available; the terminal stays open after the program exits.
    local inner="conda activate $CONDA_ENV 2>/dev/null || true; echo '$title'; echo $cmd; $cmd; echo; read -rp 'Finished. Press Enter to close...'"
    if command -v gnome-terminal >/dev/null; then
        gnome-terminal --title="$title" -- bash -ic "$inner"
    elif command -v konsole >/dev/null; then
        konsole -p tabtitle="$title" -e bash -ic "$inner" &
    elif command -v xterm >/dev/null; then
        xterm -T "$title" -e bash -ic "$inner" &
    elif command -v x-terminal-emulator >/dev/null; then
        x-terminal-emulator -e bash -ic "$inner" &
    else
        echo "No terminal emulator found (gnome-terminal, konsole, xterm)." >&2
        exit 1
    fi
}

for name in follower_pink leader_blue follower_blue leader_green; do port "$name" >/dev/null; done

open_terminal "follower_pink <- leader_blue" "$(teleop_cmd follower_pink leader_blue)"
sleep 1
open_terminal "follower_blue <- leader_green" "$(teleop_cmd follower_blue leader_green)"
