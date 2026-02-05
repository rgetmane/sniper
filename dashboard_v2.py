#!/usr/bin/env python3
"""MOLD SNIPER BOT v2.0 — FastAPI Dashboard (PRO-GRADE)"""

import os
import json
import sqlite3
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, WebSocket, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sniper.db"
LOG_PATH = BASE_DIR / "sniper.log"
MEMORY_PATH = BASE_DIR / "sniper_memory.json"
SECRETS_ENV = BASE_DIR / "secrets.env"
DOT_ENV = BASE_DIR / ".env"

# FastAPI app with CORS
app = FastAPI(title="Sniper Bot Dashboard", version="2.0-PRO")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config & Auth ────────────────────────────────────────────────────────────

API_KEY = os.getenv("DASHBOARD_API_KEY", "dev-key-change-in-prod")

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from X-API-Key header (skip in dev mode)."""
    if API_KEY == "dev-key-change-in-prod":
        return  # Dev mode: no auth required
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ── Env Loading ──────────────────────────────────────────────────────────────

def load_env_file(path):
    """Parse KEY=VALUE from env file, skip comments and blanks."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

_secrets = load_env_file(SECRETS_ENV)
_config = load_env_file(DOT_ENV)

def cfg(key, default=""):
    return _secrets.get(key, _config.get(key, os.getenv(key, default)))

# ── Wallet Address ───────────────────────────────────────────────────────────

WALLET_ADDRESS = None

def get_wallet_address():
    global WALLET_ADDRESS
    if WALLET_ADDRESS:
        return WALLET_ADDRESS
    pk = cfg("PRIVATE_KEY")
    if pk:
        try:
            import base58
            from solders.keypair import Keypair
            kp = Keypair.from_bytes(base58.b58decode(pk))
            WALLET_ADDRESS = str(kp.pubkey())
        except Exception:
            WALLET_ADDRESS = "unknown"
    else:
        WALLET_ADDRESS = "no key"
    return WALLET_ADDRESS

# ── DB Helper ────────────────────────────────────────────────────────────────

def query_db(sql, args=(), one=False):
    if not DB_PATH.exists():
        return None if one else []
    con = sqlite3.connect(str(DB_PATH), timeout=5)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(sql, args)
        rows = cur.fetchall()
        return (dict(rows[0]) if rows else None) if one else [dict(r) for r in rows]
    except Exception as e:
        print(f"DB query error: {e}")
        return None if one else []
    finally:
        con.close()

# ── Routes: API ──────────────────────────────────────────────────────────────

@app.get("/api/trades")
async def api_trades(limit: int = Query(50, le=500)):
    """Return recent token trades from the DB."""
    rows = query_db(
        "SELECT mint, symbol, score, bought, sold, txbuy, txsell, "
        "buyprice, sellprice, pnl, buytime, selltime, reason "
        "FROM tokens ORDER BY buytime DESC LIMIT ?",
        (limit,),
    )
    return {"trades": rows, "count": len(rows) if rows else 0}

@app.get("/api/logs")
async def api_logs(lines: int = Query(200, le=1000)):
    """Tail last N lines of sniper.log."""
    if not LOG_PATH.exists():
        return {"lines": [], "total": 0}
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(LOG_PATH)],
            capture_output=True, text=True, timeout=5,
        )
        log_lines = result.stdout.splitlines()
    except Exception:
        log_lines = []
    return {"lines": log_lines, "total": len(log_lines)}

@app.get("/api/wallet")
async def api_wallet():
    """Get wallet SOL balance."""
    addr = get_wallet_address()
    balance = 0.0
    error = None
    if addr and "unknown" not in addr and "no key" not in addr:
        try:
            result = subprocess.run(
                ["solana", "balance", addr, "--url", "mainnet-beta"],
                capture_output=True, text=True, timeout=5,
            )
            balance = float(result.stdout.strip().split()[0])
        except subprocess.TimeoutExpired:
            error = "RPC timeout"
        except Exception:
            helius_key = cfg("HELIUS_API_KEY")
            if helius_key:
                try:
                    import requests
                    resp = requests.post(
                        f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                        json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [addr]},
                        timeout=5,
                    )
                    lamports = resp.json().get("result", {}).get("value", 0)
                    balance = lamports / 1e9
                except Exception:
                    error = "RPC error"
            else:
                error = "no RPC"
    return {
        "address": addr,
        "balance_sol": round(balance, 6),
        "balance_lamports": int(balance * 1e9),
        "error": error,
    }

@app.get("/api/stats")
async def api_stats():
    """Aggregate stats from DB + memory file."""
    total_buys = query_db("SELECT COUNT(*) as c FROM tokens WHERE bought=1", one=True)
    total_sells = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1", one=True)
    wins = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1 AND pnl>0", one=True)
    losses = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1 AND pnl<=0", one=True)
    total_pnl = query_db("SELECT COALESCE(SUM(pnl),0) as s FROM tokens WHERE sold=1", one=True)
    blacklist_count = query_db("SELECT COUNT(*) as c FROM blacklist", one=True)

    daily_pnl = query_db(
        "SELECT date(selltime) as day, SUM(pnl) as pnl, COUNT(*) as trades "
        "FROM tokens WHERE sold=1 AND selltime IS NOT NULL "
        "GROUP BY date(selltime) ORDER BY day DESC LIMIT 30"
    )

    mem_stats = {"total_wins": 0, "total_losses": 0, "best_day_pct": 0, "rugs_caught": 0}
    if MEMORY_PATH.exists():
        try:
            mem = json.loads(MEMORY_PATH.read_text())
            s = mem.get("stats", {})
            mem_stats = {
                "total_wins": s.get("total_wins", 0),
                "total_losses": s.get("total_losses", 0),
                "best_day_pct": s.get("best_day_pct", 0),
                "rugs_caught": len(mem.get("rugs", {})),
            }
        except Exception:
            pass

    return {
        "total_buys": (total_buys or {}).get("c", 0),
        "total_sells": (total_sells or {}).get("c", 0),
        "wins": (wins or {}).get("c", 0),
        "losses": (losses or {}).get("c", 0),
        "total_pnl": round((total_pnl or {}).get("s", 0), 6),
        "blacklist_count": (blacklist_count or {}).get("c", 0),
        "daily_pnl": daily_pnl or [],
        "memory": mem_stats,
    }

@app.get("/api/status")
async def api_status():
    """Bot status and configuration."""
    running = False
    pid = None
    uptime = ""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "sniper"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().splitlines()
        if pids:
            running = True
            pid = int(pids[0])
            ps = subprocess.run(
                ["ps", "-o", "etime=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
            uptime = ps.stdout.strip()
    except Exception:
        pass

    return {
        "running": running,
        "pid": pid,
        "uptime": uptime,
        "config": {
            "max_buy_sol": cfg("MAX_BUY_SOL", "0.01"),
            "profit_target": cfg("PROFIT_TARGET_PCT", "30"),
            "stop_loss": cfg("STOP_LOSS_PCT", "10"),
            "trail_drop": cfg("TRAIL_DROP_PCT", "15"),
            "min_rug_score": cfg("MIN_RUG_SCORE", "30"),
            "jito_tip": cfg("JITO_TIP_LAMPORTS", "100000"),
            "simulate": cfg("SIMULATE_BEFORE_SEND", "true"),
        },
    }

@app.post("/api/telegram-test")
async def telegram_test(alert_type: str = "buy"):
    """Test Telegram alert."""
    import urllib.request
    token = cfg("TELEGRAM_TOKEN")
    chat_id = cfg("CHAT_ID")
    
    if not token or not chat_id:
        return {"error": "Telegram not configured", "success": False}
    
    msgs = {
        "buy": "🟢 <b>BUY TEST</b> from Dashboard",
        "sell": "✅ <b>SELL TEST</b> from Dashboard",
        "safety": "🔴 <b>SAFETY TEST</b> from Dashboard",
        "forensic": "🔥 <b>FORENSIC TEST</b> from Dashboard",
    }
    msg = msgs.get(alert_type, msgs["buy"])
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={msg}&parse_mode=HTML"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return {"success": resp.status == 200, "message": msg}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/api/risk-heatmap")
async def risk_heatmap():
    """Risk heatmap: Recent tokens with rug scores, liquidity, holder %."""
    rows = query_db(
        "SELECT mint, symbol, score, bought, reason FROM tokens ORDER BY buytime DESC LIMIT 20"
    )
    if not rows:
        return {"heatmap": []}
    
    heatmap = []
    for r in rows:
        score = int(r['score']) if r['score'] else 0
        # Color code based on rug score (0-100, higher = safer)
        if score >= 70:
            risk = "green"  # Safe
        elif score >= 40:
            risk = "yellow"  # Moderate  
        else:
            risk = "red"  # High risk
        
        heatmap.append({
            "symbol": r['symbol'],
            "mint": r['mint'][:8] + "...",
            "score": score,
            "risk": risk,
            "reason": r.get('reason', 'N/A'),
        })
    return {"heatmap": heatmap}

@app.get("/api/export-csv")
async def export_csv():
    """Export all trades as CSV."""
    rows = query_db(
        "SELECT mint, symbol, score, buyprice, sellprice, pnl, buytime, selltime, reason "
        "FROM tokens ORDER BY buytime DESC"
    )
    if not rows:
        return {"error": "No trades"}
    
    csv = "MINT,SYMBOL,SCORE,BUY_PRICE,SELL_PRICE,PNL,BUY_TIME,SELL_TIME,REASON\n"
    for r in rows:
        csv += f"{r['mint']},{r['symbol']},{r['score']},{r['buyprice']},{r['sellprice']},{r['pnl']},{r['buytime']},{r['selltime']},{r['reason']}\n"
    
    filepath = BASE_DIR / "trades_export.csv"
    filepath.write_text(csv)
    return FileResponse(filepath, media_type="text/csv", filename="trades_export.csv")

# ── WebSocket: Live Log Streaming ────────────────────────────────────────────

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """Stream sniper.log live via WebSocket."""
    await websocket.accept()
    last_pos = 0
    
    try:
        while True:
            if LOG_PATH.exists():
                with open(LOG_PATH, "r") as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                    
                    for line in new_lines:
                        await websocket.send_json({"type": "log", "line": line.rstrip()})
            
            await asyncio.sleep(0.5)  # Poll every 500ms
    except Exception:
        await websocket.close()

# ── HTML UI ──────────────────────────────────────────────────────────────────

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sniper Bot Dashboard v2.0 PRO</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --dark-bg: #0a0e27;
            --dark-card: #141829;
            --dark-border: #1f2937;
            --primary: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --text-light: #e5e7eb;
            --text-muted: #9ca3af;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--dark-bg);
            color: var(--text-light);
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        header {
            background: linear-gradient(135deg, var(--dark-card), var(--dark-bg));
            border-bottom: 1px solid var(--dark-border);
            padding: 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-flex {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        .header-right {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        button {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.2s;
            background: var(--dark-border);
            color: var(--text-light);
        }
        
        button:hover {
            background: var(--primary);
            color: #000;
        }
        
        .btn-danger:hover { background: var(--danger); }
        .btn-warning:hover { background: var(--warning); }
        
        .toggle-dark {
            width: 2.5rem;
            height: 1.3rem;
            background: var(--dark-border);
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }
        
        .toggle-dark.active {
            background: var(--primary);
        }
        
        .toggle-dark::after {
            content: '';
            position: absolute;
            width: 1rem;
            height: 1rem;
            background: white;
            border-radius: 50%;
            top: 0.15rem;
            left: 0.15rem;
            transition: all 0.3s;
        }
        
        .toggle-dark.active::after {
            left: 1.35rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        @media (min-width: 768px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
            .grid.wide { grid-template-columns: repeat(3, 1fr); }
        }
        
        @media (min-width: 1200px) {
            .grid { grid-template-columns: repeat(3, 1fr); }
            .grid.wide { grid-template-columns: repeat(4, 1fr); }
        }
        
        .card {
            background: var(--dark-card);
            border: 1px solid var(--dark-border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            transition: all 0.2s;
        }
        
        .card:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
        }
        
        .card h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        .card-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        
        .card-meta {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        .status-indicator {
            display: inline-block;
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        .status-online { background: var(--primary); }
        .status-offline { background: var(--danger); }
        
        .chart-wrapper {
            background: var(--dark-card);
            border: 1px solid var(--dark-border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .logs-panel {
            background: var(--dark-card);
            border: 1px solid var(--dark-border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.85rem;
        }
        
        .log-line {
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            color: var(--text-muted);
        }
        
        .log-line:last-child { border-bottom: none; }
        .log-line.warn { color: var(--warning); }
        .log-line.error { color: var(--danger); }
        .log-line.success { color: var(--primary); }
        
        .safety-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        
        .safety-item {
            background: var(--dark-bg);
            border: 1px solid var(--dark-border);
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .safety-item.active {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.1);
        }
        
        .safety-item.inactive {
            border-color: var(--danger);
            background: rgba(239, 68, 68, 0.1);
        }
        
        .safety-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            margin-top: 0.5rem;
            color: var(--text-muted);
        }
        
        .tabs {
            display: flex;
            gap: 0;
            border-bottom: 1px solid var(--dark-border);
            margin-bottom: 1.5rem;
        }
        
        .tab {
            padding: 1rem 1.5rem;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }
        
        .tab:hover { color: var(--text-light); }
        
        @media (max-width: 640px) {
            .grid { grid-template-columns: 1fr; }
            .header-flex {
                flex-direction: column;
                gap: 1rem;
            }
            .card-value { font-size: 1.5rem; }
            .container { padding: 1rem 0.5rem; }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-flex">
            <h1>🔥 Sniper Bot v2.0 PRO</h1>
            <div class="header-right">
                <button id="btn-refresh" onclick="refreshAll()">↻ Refresh</button>
                <button id="btn-telegram" onclick="testTelegram()" class="btn-warning">📱 Telegram</button>
                <button id="btn-export" onclick="exportCSV()">📥 Export</button>
                <div class="toggle-dark" id="theme-toggle"></div>
            </div>
        </div>
    </header>
    
    <div class="container">
        <!-- STATUS CARDS -->
        <div class="grid wide">
            <div class="card">
                <h3>Bot Status</h3>
                <div id="bot-status" style="font-size: 2.5rem;">○</div>
                <div class="card-meta" id="bot-meta">Initializing...</div>
            </div>
            <div class="card">
                <h3>Wallet Balance</h3>
                <div class="card-value" id="wallet-balance">0.00 SOL</div>
                <div class="card-meta" id="wallet-address">Loading...</div>
            </div>
            <div class="card">
                <h3>Total P&L</h3>
                <div class="card-value" id="total-pnl">0.00 SOL</div>
                <div class="card-meta"><span id="win-rate">0/0</span> trades</div>
            </div>
        </div>

        <!-- STATS GRID -->
        <div class="grid">
            <div class="card">
                <h3>Buys</h3>
                <div class="card-value" id="stat-buys">0</div>
            </div>
            <div class="card">
                <h3>Sells</h3>
                <div class="card-value" id="stat-sells">0</div>
            </div>
            <div class="card">
                <h3>Wins</h3>
                <div class="card-value" id="stat-wins">0</div>
            </div>
        </div>

        <!-- CHART -->
        <div class="chart-wrapper">
            <h3 style="margin-bottom: 1rem;">Daily P&L</h3>
            <canvas id="pnl-chart" height="80"></canvas>
        </div>

        <!-- SAFETY GUARDS - 8 LAYER PROTECTION -->
        <div class="card" style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem;">🛡️ 8-Layer Safety System</h3>
            <div class="safety-grid">
                <div class="safety-item active" title="Rug score minimum 30">
                    <span style="font-size: 1.5rem;">①</span>
                    <div class="safety-label">Min Rug Score</div>
                </div>
                <div class="safety-item active" title="Max top holder 25%">
                    <span style="font-size: 1.5rem;">②</span>
                    <div class="safety-label">Holder %</div>
                </div>
                <div class="safety-item active" title="Min liquidity 5 SOL">
                    <span style="font-size: 1.5rem;">③</span>
                    <div class="safety-label">Liquidity</div>
                </div>
                <div class="safety-item active" title="Mint authority check">
                    <span style="font-size: 1.5rem;">④</span>
                    <div class="safety-label">Mint Auth</div>
                </div>
                <div class="safety-item active" title="Freeze authority check">
                    <span style="font-size: 1.5rem;">⑤</span>
                    <div class="safety-label">Freeze Auth</div>
                </div>
                <div class="safety-item active" title="Max 3 failed buys before lock">
                    <span style="font-size: 1.5rem;">⑥</span>
                    <div class="safety-label">Buy Limit</div>
                </div>
                <div class="safety-item active" title="2s delay between attempts">
                    <span style="font-size: 1.5rem;">⑦</span>
                    <div class="safety-label">Buy Delay</div>
                </div>
                <div class="safety-item active" title="Program execution whitelist">
                    <span style="font-size: 1.5rem;">⑧</span>
                    <div class="safety-label">Whitelist</div>
                </div>
            </div>
        </div>

        <!-- RISK HEATMAP -->
        <div class="card" style="margin-bottom: 2rem;">
            <h3 style="margin-bottom: 1rem;">🔥 Token Risk Analysis</h3>
            <div id="heatmap-container" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 0.5rem;">
                <div style="text-align: center; color: var(--text-muted);">Loading...</div>
            </div>
        </div>

        <!-- TABS: LOGS & TRADES -->
        <div class="tabs">
            <button class="tab active" onclick="switchTab('logs')">📋 Live Logs</button>
            <button class="tab" onclick="switchTab('trades')">💰 Recent Trades</button>
        </div>

        <div id="tab-logs" class="logs-panel">
            <div id="logs-container"></div>
        </div>

        <div id="tab-trades" style="display:none;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 2px solid var(--dark-border);">
                        <th style="padding: 0.75rem; text-align: left;">Token</th>
                        <th style="padding: 0.75rem; text-align: right;">Entry</th>
                        <th style="padding: 0.75rem; text-align: right;">Exit</th>
                        <th style="padding: 0.75rem; text-align: right;">P&L</th>
                    </tr>
                </thead>
                <tbody id="trades-tbody"></tbody>
            </table>
        </div>
    </div>

    <script>
        const API = '/api';
        let chartInstance = null;
        let wsLogs = null;

        // Auto-refresh every 5s
        setInterval(refreshAll, 5000);

        function refreshAll() {
            fetch(`${API}/status`).then(r => r.json()).then(data => {
                const status = data.running ? '🟢 LIVE' : '⚫ OFFLINE';
                const indicator = data.running ? 'status-online' : 'status-offline';
                document.getElementById('bot-status').innerHTML = 
                    `<span class="status-indicator ${indicator}"></span>${status}`;
                document.getElementById('bot-meta').innerHTML = 
                    data.uptime ? `Uptime: ${data.uptime}` : 'Not running';
            });

            fetch(`${API}/wallet`).then(r => r.json()).then(data => {
                document.getElementById('wallet-balance').textContent = `${data.balance_sol} SOL`;
                document.getElementById('wallet-address').textContent = 
                    data.address.substring(0, 8) + '...' + data.address.substring(-8);
            });

            fetch(`${API}/stats`).then(r => r.json()).then(data => {
                document.getElementById('stat-buys').textContent = data.total_buys;
                document.getElementById('stat-sells').textContent = data.total_sells;
                document.getElementById('stat-wins').textContent = data.wins;
                document.getElementById('total-pnl').textContent = `${data.total_pnl.toFixed(4)} SOL`;
                document.getElementById('win-rate').textContent = `${data.wins}/${data.total_sells}`;
                
                // Update chart
                updateChart(data.daily_pnl);
            });

            fetch(`${API}/trades?limit=10`).then(r => r.json()).then(data => {
                const tbody = document.getElementById('trades-tbody');
                tbody.innerHTML = data.trades.map(t => `
                    <tr style="border-bottom: 1px solid var(--dark-border);">
                        <td style="padding: 0.75rem;">${t.symbol || t.mint.substring(0, 8)}</td>
                        <td style="padding: 0.75rem; text-align: right;">${parseFloat(t.buyprice).toFixed(6)}</td>
                        <td style="padding: 0.75rem; text-align: right;">${t.sellprice ? parseFloat(t.sellprice).toFixed(6) : '—'}</td>
                        <td style="padding: 0.75rem; text-align: right; color: ${t.pnl > 0 ? 'var(--primary)' : 'var(--danger)'};">
                            ${t.pnl ? `${(t.pnl > 0 ? '+' : '')}${t.pnl.toFixed(6)}` : '—'}
                        </td>
                    </tr>
                `).join('');
            });
        }

        function updateChart(data) {
            if (!data || data.length === 0) return;
            const ctx = document.getElementById('pnl-chart').getContext('2d');
            const labels = data.map(d => d.day).reverse();
            const values = data.map(d => d.pnl).reverse();
            
            if (chartInstance) chartInstance.destroy();
            
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Daily P&L (SOL)',
                        data: values,
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            ticks: { color: 'rgba(229, 231, 235, 0.7)' },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        },
                        x: {
                            ticks: { color: 'rgba(229, 231, 235, 0.7)' },
                            grid: { display: false },
                        }
                    }
                }
            });
        }

        function switchTab(tab) {
            ['logs', 'trades'].forEach(t => {
                document.getElementById(`tab-${t}`).style.display = tab === t ? 'block' : 'none';
            });
            document.querySelectorAll('.tab').forEach((el, i) => {
                el.classList.toggle('active', (i === 0 && tab === 'logs') || (i === 1 && tab === 'trades'));
            });
        }

        function testTelegram() {
            const btn = document.getElementById('btn-telegram');
            btn.textContent = '⏳ Sending...';
            btn.disabled = true;
            fetch(`${API}/telegram-test?alert_type=buy`, { method: 'POST' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        btn.textContent = '✅ Sent!';
                        setTimeout(() => {
                            btn.textContent = '📱 Telegram';
                            btn.disabled = false;
                        }, 3000);
                    } else {
                        btn.textContent = '❌ Error';
                        btn.disabled = false;
                        setTimeout(() => {
                            btn.textContent = '📱 Telegram';
                        }, 2000);
                    }
                })
                .catch(() => {
                    btn.textContent = '❌ Error';
                    btn.disabled = false;
                });
        }

        function loadHeatmap() {
            fetch(`${API}/risk-heatmap`).then(r => r.json()).then(data => {
                const container = document.getElementById('heatmap-container');
                if (!data.heatmap || data.heatmap.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: var(--text-muted);">No tokens scanned yet</div>';
                    return;
                }
                
                container.innerHTML = data.heatmap.map(token => {
                    let bgColor = 'rgba(34, 197, 94, 0.2)';
                    let borderColor = 'rgb(34, 197, 94)';
                    
                    if (token.risk === 'yellow') {
                        bgColor = 'rgba(245, 158, 11, 0.2)';
                        borderColor = 'rgb(245, 158, 11)';
                    } else if (token.risk === 'red') {
                        bgColor = 'rgba(239, 68, 68, 0.2)';
                        borderColor = 'rgb(239, 68, 68)';
                    }
                    
                    return `
                        <div style="
                            padding: 0.75rem;
                            border: 2px solid ${borderColor};
                            border-radius: 0.5rem;
                            background: ${bgColor};
                            text-align: center;
                            cursor: pointer;
                            transition: all 0.2s;
                        " title="${token.symbol} - Score: ${token.score}">
                            <div style="font-weight: 600; font-size: 0.85rem;">${token.symbol}</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">
                                Score: ${token.score}
                            </div>
                        </div>
                    `;
                }).join('');
            });
        }

        function exportCSV() {
            window.location = `${API}/export-csv`;
        }

        // Live logs via WebSocket
        function connectLogs() {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            wsLogs = new WebSocket(`${proto}//${window.location.host}/ws/logs`);
            wsLogs.onmessage = (e) => {
                const data = JSON.parse(e.data);
                const container = document.getElementById('logs-container');
                const line = document.createElement('div');
                line.className = 'log-line';
                if (data.line.includes('SAFETY: REJECTED') || data.line.includes('ERROR')) {
                    line.classList.add('error');
                } else if (data.line.includes('PASSED') || data.line.includes('EXECUTED')) {
                    line.classList.add('success');
                }
                line.textContent = data.line;
                container.appendChild(line);
                container.scrollTop = container.scrollHeight;
                // Keep only last 100 lines
                while (container.children.length > 100) {
                    container.removeChild(container.firstChild);
                }
            };
        }

        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', function() {
            document.body.classList.toggle('light-mode');
            this.classList.toggle('active');
            localStorage.setItem('theme', this.classList.contains('active') ? 'dark' : 'light');
        });

        // Init
        refreshAll();
        loadHeatmap();
        connectLogs();
        
        // Refresh heatmap every 10s
        setInterval(loadHeatmap, 10000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_UI

# ── Easter Egg ───────────────────────────────────────────────────────────────

@app.get("/admin/rpc-calls")
async def admin_rpc():
    """Easter egg: Show raw RPC calls from log"""
    if not LOG_PATH.exists():
        return {"rpc_calls": []}
    
    with open(LOG_PATH) as f:
        lines = f.readlines()
    
    rpc_calls = [l.strip() for l in lines if 'RPC:' in l or 'sendTransaction' in l]
    return {"rpc_calls": rpc_calls[-20:], "total": len(rpc_calls)}

# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", 5000))
    print(f"\n  🚀 SNIPER DASHBOARD v2.0 PRO")
    print(f"  http://localhost:{port}")
    print(f"  WebSocket: ws://localhost:{port}/ws/logs\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
