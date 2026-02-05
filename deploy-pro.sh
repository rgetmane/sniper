#!/bin/bash
# Dashboard Pro Deployment Script
# Installs Caddy, creates self-signed cert, and runs dashboard with HTTPS

set -e

SNIPER_DIR="/home/roman/sniper"
DASHBOARD_API_KEY="${DASHBOARD_API_KEY:-change-me-in-prod}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5000}"
CADDY_PORT="${CADDY_PORT:-5443}"

echo "🚀 SNIPER DASHBOARD v2.0 PRO - DEPLOYMENT"
echo "==========================================="

# Install Caddy if not present
if ! command -v caddy &> /dev/null; then
    echo "📥 Installing Caddy..."
    cd /tmp
    wget -q https://github.com/caddyserver/caddy/releases/download/v2.7.6/caddy_2.7.6_linux_amd64.tar.gz -O caddy.tar.gz
    tar -xzf caddy.tar.gz caddy
    sudo mv caddy /usr/local/bin/
    rm caddy.tar.gz
    echo "✅ Caddy installed"
else
    echo "✅ Caddy already installed"
fi

# Kill existing dashboard
pkill -f "python3 dashboard_v2.py" 2>/dev/null || true
sleep 1

# Create Caddyfile
cat > "$SNIPER_DIR/Caddyfile" << 'EOF'
localhost:5443 {
    tls internal
    
    reverse_proxy localhost:5000 {
        header_up X-Real-IP {http.request.remote}
        header_up X-Forwarded-For {http.request.remote}
        header_up X-Forwarded-Proto {http.request.proto}
    }
    
    # WebSocket support
    header Connection "upgrade"
    header Upgrade "websocket"
}
EOF

echo "✅ Caddyfile created at $SNIPER_DIR/Caddyfile"

# Start FastAPI dashboard (internal port 5000)
echo "🟢 Starting FastAPI dashboard on localhost:5000..."
cd "$SNIPER_DIR"
nohup python3 dashboard_v2.py > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 2

# Start Caddy (public port 5443 with HTTPS)
echo "🔒 Starting Caddy reverse proxy with auto-cert..."
nohup caddy run --config "$SNIPER_DIR/Caddyfile" > /tmp/caddy.log 2>&1 &
CADDY_PID=$!
sleep 2

echo ""
echo "✅ DASHBOARD LIVE!"
echo "==========================================="
echo "🌐 HTTPS: https://localhost:5443"
echo "📱 API: http://localhost:5000/api/"
echo "📡 WebSocket: wss://localhost:5443/ws/logs"
echo ""
echo "🔑 API Key (add to header): X-API-Key: $DASHBOARD_API_KEY"
echo ""
echo "📊 Status: Both dashboard ($DASHBOARD_PID) and caddy ($CADDY_PID) running"
echo "📝 Logs: tail -f /tmp/dashboard.log & tail -f /tmp/caddy.log"
echo "==========================================="
