#!/bin/bash
# Auto-restart wrapper for sniper bot

cd "$(dirname "$0")"

while true; do
    echo "[$(date)] Starting sniper..."
    ./sniper >> sniper.log 2>&1
    EXIT_CODE=$?
    echo "[$(date)] Sniper exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Clean exit, stopping"
        break
    fi

    echo "[$(date)] Crash detected, restarting in 5 seconds..."
    sleep 5
done
