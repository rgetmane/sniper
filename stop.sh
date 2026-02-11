#!/bin/bash
# Stop script for sniper bot and watchdog

cd /home/roman/sniper

echo "Stopping sniper bot..."

# Kill watchdog first
pkill -9 -f watchdog.sh
echo "✓ Watchdog stopped"

# Kill sniper
pkill -9 sniper
echo "✓ Sniper stopped"

sleep 1

# Verify
if pgrep -f "(sniper|watchdog)" > /dev/null; then
    echo "⚠️  Some processes still running:"
    ps aux | grep -E "(sniper|watchdog)" | grep -v grep
else
    echo "✅ All processes stopped"
fi
