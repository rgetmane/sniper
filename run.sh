#!/bin/bash
cd "$(dirname "$0")"

export $(grep -v '^#' secrets.env | xargs)
export $(grep -v '^#' .env | xargs)

LOG="mold.log"
PID="mold.pid"

cleanup() {
    echo "[$(date)] Shutdown signal received" >> "$LOG"
    [ -f "$PID" ] && kill $(cat "$PID") 2>/dev/null
    rm -f "$PID"
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    echo "[$(date)] Starting mold sniper..." >> "$LOG"

    ./mold >> "$LOG" 2>&1 &
    echo $! > "$PID"

    wait $(cat "$PID")
    CODE=$?

    echo "[$(date)] Mold exited with code $CODE" >> "$LOG"

    [ $CODE -eq 0 ] && { echo "[$(date)] Clean exit" >> "$LOG"; rm -f "$PID"; break; }

    echo "[$(date)] Crash detected, restarting in 5s..." >> "$LOG"
    sleep 5
done
