#!/bin/bash
# Watchdog script for sniper bot
# Monitors and auto-restarts the bot if it crashes

cd /home/roman/sniper

PROCESS_NAME="sniper"
LOG_FILE="watchdog.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log "Watchdog started - monitoring $PROCESS_NAME every 30s"

while true; do
    # Check if process is running
    if pgrep -f "./$PROCESS_NAME" > /dev/null; then
        # Process is alive
        sleep 30
    else
        # Process is dead - restart it
        log "⚠️  Process died - restarting $PROCESS_NAME"
        
        # Kill any zombie processes
        pkill -9 "$PROCESS_NAME" 2>/dev/null
        sleep 1
        
        # Start fresh
        nohup ./$PROCESS_NAME > sniper.log 2>&1 &
        NEW_PID=$!
        
        log "✓ Restarted $PROCESS_NAME (PID: $NEW_PID) – uptime reset"
        
        # Send Telegram alert
        if [ -f secrets.env ]; then
            source secrets.env
            if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$CHAT_ID" ]; then
                curl -s "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage?chat_id=${CHAT_ID}&text=⚠️%20Sniper%20auto-restarted%20by%20watchdog" > /dev/null
            fi
        fi
        
        sleep 30
    fi
done
