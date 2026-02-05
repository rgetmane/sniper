#!/bin/bash
# DigitalOcean Auto-Deploy Script for Sniper Bot

set -e

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Configuration
DO_USER=${DO_USER:-"root"}
DO_HOST="206.81.4.22"
DO_KEY_PATH=${DO_KEY_PATH:-"~/.ssh/id_ed25519"}
REMOTE_DIR="/home/${DO_USER}/sniper"

echo "========================================="
echo "  DIGITALOCEAN DEPLOYMENT"
echo "========================================="
echo "Target: ${DO_USER}@${DO_HOST}"
echo "Remote: ${REMOTE_DIR}"
echo

# Test SSH connection
echo "Testing SSH connection..."
if ! ssh -i "$DO_KEY_PATH" -o ConnectTimeout=5 -o BatchMode=yes "${DO_USER}@${DO_HOST}" "echo 'SSH OK'" 2>/dev/null; then
    echo "❌ SSH connection failed"
    echo "Check:"
    echo "  1. SSH key exists: $DO_KEY_PATH"
    echo "  2. Key added to DO: ssh-copy-id -i $DO_KEY_PATH ${DO_USER}@${DO_HOST}"
    echo "  3. Server is reachable: ping $DO_HOST"
    exit 1
fi
echo "✓ SSH connection OK"

# Create remote directory
echo "Creating remote directory..."
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "mkdir -p ${REMOTE_DIR}"
echo "✓ Remote directory ready"

# Stop remote bot
echo "Stopping remote bot..."
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "pkill -9 sniper 2>/dev/null || true; pkill -9 watchdog 2>/dev/null || true"
echo "✓ Remote bot stopped"

# Upload files
echo "Uploading files..."
scp -i "$DO_KEY_PATH" -r \
    main.go \
    go.mod \
    go.sum \
    .env \
    secrets.env \
    start.sh \
    stop.sh \
    watchdog.sh \
    "${DO_USER}@${DO_HOST}:${REMOTE_DIR}/" 2>&1 | grep -v "Permanently added"

echo "✓ Files uploaded"

# Build and start on remote
echo "Building on remote server..."
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "cd ${REMOTE_DIR} && go build -o sniper main.go && chmod +x *.sh"
echo "✓ Build complete"

echo "Starting remote bot..."
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "cd ${REMOTE_DIR} && nohup ./sniper > sniper.log 2>&1 &"
sleep 2

# Start watchdog
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "cd ${REMOTE_DIR} && nohup ./watchdog.sh > /dev/null 2>&1 &"
echo "✓ Bot and watchdog started"

# Check status
echo
echo "Checking remote status..."
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "ps aux | grep -E '(sniper|watchdog)' | grep -v grep" || echo "No processes found (may still be starting)"

# Show logs
echo
echo "========================================="
echo "  REMOTE LOGS (last 15 lines)"
echo "========================================="
ssh -i "$DO_KEY_PATH" "${DO_USER}@${DO_HOST}" "cd ${REMOTE_DIR} && tail -15 sniper.log"

echo
echo "========================================="
echo "✅ Deployment complete!"
echo "========================================="
echo "Monitor: ssh -i $DO_KEY_PATH ${DO_USER}@${DO_HOST} 'tail -f ${REMOTE_DIR}/sniper.log'"
echo "Stop:    ssh -i $DO_KEY_PATH ${DO_USER}@${DO_HOST} 'cd ${REMOTE_DIR} && ./stop.sh'"
echo "========================================="
