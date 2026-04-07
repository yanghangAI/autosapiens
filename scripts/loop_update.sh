#!/bin/bash
# Runs update-all every minute in a loop.
# Start in a persistent tmux session:
#   tmux new -s autoupdate
#   bash scripts/loop_update.sh
#   Ctrl+B, D  to detach

REPO=/work/pi_nwycoff_umass_edu/hang/auto
LOG=$REPO/cron_hook.log
INTERVAL=60

while true; do
    cd "$REPO"
    /usr/bin/python3 scripts/cli.py update-all >> "$LOG" 2>&1
    sleep $INTERVAL
done
