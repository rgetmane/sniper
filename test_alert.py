#!/usr/bin/env python3
"""Test Telegram alerts for sniper bot."""

import os
import json
import urllib.request
import time
from datetime import datetime

def load_env():
    """Load secrets.env variables."""
    env = {}
    if os.path.exists('secrets.env'):
        with open('secrets.env') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                env[key.strip()] = val.strip()
    return env

def send_telegram(token, chat_id, message):
    """Send Telegram message with retry logic."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    
    for attempt in range(3):
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.load(resp)
                if result.get("ok"):
                    return True, f"✅ Sent (attempt {attempt+1})"
                else:
                    return False, f"❌ API error: {result.get('description')}"
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1}... ({str(e)[:40]})")
                time.sleep(1 + attempt)
            else:
                return False, f"❌ Failed after 3 attempts: {e}"
    
    return False, "❌ Unknown error"

def main():
    env = load_env()
    token = env.get('TELEGRAM_TOKEN')
    chat_id = env.get('CHAT_ID')
    
    if not token or not chat_id:
        print("❌ TELEGRAM_TOKEN or CHAT_ID not set in secrets.env")
        return
    
    print("=" * 60)
    print("  SNIPER BOT TELEGRAM ALERT TEST")
    print("=" * 60)
    print(f"\nToken: {token[:20]}...{token[-5:]}")
    print(f"Chat ID: {chat_id}")
    print("\nSending test alerts...\n")
    
    # Test 1: BUY alert
    buy_msg = (
        "🟢 <b>BUY EXECUTED</b>\n"
        "Token: TokenSymbolXYZ (SYMBOL)\n"
        "Entry: 0.01 SOL\n"
        "Risk Score: 850\n"
        "Link: https://solscan.io/token/TOKEN_MINT"
    )
    print("[1/5] BUY alert...")
    success, msg = send_telegram(token, chat_id, buy_msg)
    print(f"     {msg}\n")
    time.sleep(2)
    
    # Test 2: SAFETY BLOCK alert
    safety_msg = (
        "🔴 <b>RUG BLOCKED</b> — Safety Guard #3 Engaged\n"
        "Token: SuspiciousMint123\n"
        "Reason: Top holder 95% > 25%\n"
        "Action: Added to blacklist"
    )
    print("[2/5] Safety block alert...")
    success, msg = send_telegram(token, chat_id, safety_msg)
    print(f"     {msg}\n")
    time.sleep(2)
    
    # Test 3: SELL/PROFIT alert
    sell_msg = (
        "✅ <b>SELL — PROFIT LOCKED</b>\n"
        "Token: TokenSymbolXYZ\n"
        "Exit: 0.01342 SOL (+34.2%)\n"
        "P&L: +0.00342 SOL\n"
        "Link: https://solscan.io/token/TOKEN_MINT"
    )
    print("[3/5] SELL/Profit alert...")
    success, msg = send_telegram(token, chat_id, sell_msg)
    print(f"     {msg}\n")
    time.sleep(2)
    
    # Test 4: LOW BALANCE alert
    low_bal_msg = (
        "⚠️ <b>LOW BALANCE WARNING</b>\n"
        "Current: 0.019 SOL\n"
        "Minimum: 0.020 SOL\n"
        "Action: Bot will STOP trading"
    )
    print("[4/5] Low balance alert...")
    success, msg = send_telegram(token, chat_id, low_bal_msg)
    print(f"     {msg}\n")
    time.sleep(2)
    
    # Test 5: FORENSIC LOCK alert
    lock_msg = (
        "🔥 <b>FORENSIC LOCK ENGAGED</b>\n"
        "Failed buy attempts: 3/3\n"
        "Action: Bot halted for safety\n"
        "Status: All positions closed, wallet secured"
    )
    print("[5/5] Forensic lock alert...")
    success, msg = send_telegram(token, chat_id, lock_msg)
    print(f"     {msg}\n")
    
    print("=" * 60)
    print("✅ ALL TEST ALERTS COMPLETED")
    print("=" * 60)
    print("\nIf you received 5 messages on Telegram, all alerts are working!")
    print("Check your phone now → should have 5 notifications\n")

if __name__ == '__main__':
    main()
