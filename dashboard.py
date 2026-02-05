#!/usr/bin/env python3
"""MOLD SNIPER BOT v2.0 — Dashboard"""

import os
import json
import sqlite3
import subprocess
from pathlib import Path
from collections import deque

from flask import Flask, render_template, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sniper.db"
LOG_PATH = BASE_DIR / "sniper.log"
MEMORY_PATH = BASE_DIR / "sniper_memory.json"
SECRETS_ENV = BASE_DIR / "secrets.env"
DOT_ENV = BASE_DIR / ".env"

app = Flask(__name__)

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
            WALLET_ADDRESS = "unknown (install solana-py)"
    else:
        WALLET_ADDRESS = "no key configured"
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
        app.logger.warning(f"DB query error: {e}")
        return None if one else []
    finally:
        con.close()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trades")
def api_trades():
    """Return recent token trades from the DB."""
    limit = request.args.get("limit", 50, type=int)
    rows = query_db(
        "SELECT mint, symbol, score, bought, sold, txbuy, txsell, "
        "buyprice, sellprice, pnl, buytime, selltime, reason "
        "FROM tokens ORDER BY buytime DESC LIMIT ?",
        (limit,),
    )
    return jsonify(rows)


@app.route("/api/logs")
def api_logs():
    """Tail last N lines of sniper.log."""
    n = request.args.get("lines", 200, type=int)
    n = min(n, 1000)
    if not LOG_PATH.exists():
        return jsonify({"lines": [], "total": 0})
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(LOG_PATH)],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.splitlines()
    except Exception:
        lines = []
    return jsonify({"lines": lines, "total": len(lines)})


@app.route("/api/wallet")
def api_wallet():
    """Get wallet SOL balance — tries solana CLI first, falls back to Helius RPC."""
    addr = get_wallet_address()
    balance = 0.0
    error = None
    if addr and "unknown" not in addr and "no key" not in addr:
        # Try solana CLI (5s timeout so it never hangs)
        try:
            result = subprocess.run(
                ["solana", "balance", addr, "--url", "mainnet-beta"],
                capture_output=True, text=True, timeout=5,
            )
            balance = float(result.stdout.strip().split()[0])
        except subprocess.TimeoutExpired:
            error = "RPC timeout"
        except Exception:
            # Fallback: Helius JSON-RPC (also 5s timeout)
            helius_key = cfg("HELIUS_API_KEY")
            if helius_key:
                try:
                    import requests as req
                    resp = req.post(
                        f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                        json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [addr]},
                        timeout=5,
                    )
                    lamports = resp.json().get("result", {}).get("value", 0)
                    balance = lamports / 1e9
                except Exception:
                    error = "RPC timeout"
            else:
                error = "no RPC configured"
    return jsonify({
        "address": addr,
        "balance_sol": round(balance, 6),
        "balance_lamports": int(balance * 1e9),
        "error": error,
    })


@app.route("/api/stats")
def api_stats():
    """Aggregate stats from DB + memory file."""
    # DB stats
    total_buys = query_db("SELECT COUNT(*) as c FROM tokens WHERE bought=1", one=True)
    total_sells = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1", one=True)
    wins = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1 AND pnl>0", one=True)
    losses = query_db("SELECT COUNT(*) as c FROM tokens WHERE sold=1 AND pnl<=0", one=True)
    total_pnl = query_db("SELECT COALESCE(SUM(pnl),0) as s FROM tokens WHERE sold=1", one=True)
    blacklist_count = query_db("SELECT COUNT(*) as c FROM blacklist", one=True)

    # Daily P&L for chart
    daily_pnl = query_db(
        "SELECT date(selltime) as day, SUM(pnl) as pnl, COUNT(*) as trades "
        "FROM tokens WHERE sold=1 AND selltime IS NOT NULL "
        "GROUP BY date(selltime) ORDER BY day DESC LIMIT 30"
    )

    # Memory stats
    mem_stats = {"total_wins": 0, "total_losses": 0, "best_day_pct": 0, "rugs_caught": 0}
    if MEMORY_PATH.exists():
        try:
            mem = json.loads(MEMORY_PATH.read_text())
            s = mem.get("stats", {})
            mem_stats["total_wins"] = s.get("total_wins", 0)
            mem_stats["total_losses"] = s.get("total_losses", 0)
            mem_stats["best_day_pct"] = s.get("best_day_pct", 0)
            mem_stats["rugs_caught"] = len(mem.get("rugs", {}))
        except Exception:
            pass

    return jsonify({
        "total_buys": (total_buys or {}).get("c", 0),
        "total_sells": (total_sells or {}).get("c", 0),
        "wins": (wins or {}).get("c", 0),
        "losses": (losses or {}).get("c", 0),
        "total_pnl": round((total_pnl or {}).get("s", 0), 6),
        "blacklist_count": (blacklist_count or {}).get("c", 0),
        "daily_pnl": daily_pnl or [],
        "memory": mem_stats,
    })


@app.route("/api/status")
def api_status():
    """Check if sniper bot process is running."""
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
            # Get uptime
            ps = subprocess.run(
                ["ps", "-o", "etime=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
            uptime = ps.stdout.strip()
    except Exception:
        pass

    return jsonify({
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
    })


if __name__ == "__main__":
    print(f"\n  MOLD SNIPER DASHBOARD")
    print(f"  http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
