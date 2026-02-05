#!/bin/bash
# Start script for sniper bot with watchdog

cd /home/roman/sniper

echo "========================================="
echo "    SNIPER BOT LAUNCHER v2.0"
echo "========================================="
echo

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -9 sniper 2>/dev/null
pkill -9 watchdog 2>/dev/null
sleep 1

# Check environment
echo "Checking environment..."
if [ ! -f secrets.env ]; then
    echo "❌ ERROR: secrets.env not found"
    exit 1
fi

source secrets.env

if [ -z "$PRIVATE_KEY" ] || [ -z "$TELEGRAM_TOKEN" ]; then
    echo "❌ ERROR: Missing required environment variables"
    exit 1
fi

echo "✓ Environment OK"

# Check if binary exists
if [ ! -f ./sniper ]; then
    echo "Building sniper..."
    go build -o sniper main.go
    if [ $? -ne 0 ]; then
        echo "❌ Build failed"
        exit 1
    fi
    echo "✓ Build OK"
fi

# Start the bot
echo
echo "Starting sniper bot..."
nohup ./sniper > sniper.log 2>&1 &
SNIPER_PID=$!
echo "✓ Sniper started (PID: $SNIPER_PID)"

# Wait a moment for startup
sleep 2

# Check if it's still running
if ! ps -p $SNIPER_PID > /dev/null; then
    echo "❌ Sniper failed to start - check sniper.log"
    tail -20 sniper.log
    exit 1
fi

# Start watchdog
echo "Starting watchdog..."
nohup ./watchdog.sh > /dev/null 2>&1 &
WATCHDOG_PID=$!
echo "✓ Watchdog started (PID: $WATCHDOG_PID)"

echo
echo "========================================="
echo "✅ Bot locked and loaded!"
echo "========================================="
echo "Sniper PID:   $SNIPER_PID"
echo "Watchdog PID: $WATCHDOG_PID"
echo
echo "Monitor: tail -f sniper.log"
echo "Stop:    ./stop.sh"
echo "========================================="
