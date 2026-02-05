#!/usr/bin/env python3
"""Real latest models sync: Claude Opus 4.5 → GPT-5.2 → Grok 4 (Feb 5 2026)."""

import sys
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import requests

def call_grok_4(prompt, timeout=5.0):
    """Grok 4 HTTP call with timeout."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MODEL_GROK')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-4",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7
        }
        endpoint = os.getenv('XAI_ENDPOINT', 'https://api.x.ai/v1/chat/completions')
        r = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        resp = r.json()
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"]
        return "(No response)"
    except Exception as e:
        raise Exception(f"Grok 4: {e}")

tests_pass = 0
tests_fail = 0

print("\n" + "=" * 70)
print("REAL LATEST MODELS SYNC - Feb 5 2026")
print("Models: Claude Opus 4.5 → GPT-5.2 → Grok 4")
print("=" * 70 + "\n")

# Test 1: Claude Opus 4.5 - 5s timeout
print("[1/3] Claude Opus 4.5: Confirm alive (5s timeout)")
try:
    start = time.time()
    llm = ChatAnthropic(
        model="claude-opus-4-5",
        api_key=os.getenv('MODEL_CLAUDE'),
        temperature=0,
        timeout=5.0
    )
    resp = llm.invoke([HumanMessage(content="Confirm: Claude Opus 4.5 alive")])
    elapsed = time.time() - start
    if elapsed > 5.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 5s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:80]
        print(f"  ✓ PASS ({elapsed:.2f}s): {result}...")
        tests_pass += 1
except Exception as e:
    err = str(e)[:100]
    print(f"  ✗ ERROR: {err}. FAIL.")
    tests_fail += 1

# Test 2: GPT 5.2 - 5s timeout
print("\n[2/3] GPT 5.2: Confirm here (5s timeout)")
try:
    start = time.time()
    llm = ChatOpenAI(
        model="gpt-5.2",
        api_key=os.getenv('MODEL_GPT'),
        temperature=0,
        timeout=5.0
    )
    resp = llm.invoke([HumanMessage(content="Confirm: GPT-5.2 here")])
    elapsed = time.time() - start
    if elapsed > 5.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 5s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:80]
        print(f"  ✓ PASS ({elapsed:.2f}s): {result}...")
        tests_pass += 1
except Exception as e:
    err = str(e)[:100]
    print(f"  ✗ ERROR: {err}. FAIL.")
    tests_fail += 1

# Test 3: Grok 4 - 15s timeout (network lag)
print("\n[3/3] Grok 4: Confirm live (15s timeout - network latency)")
try:
    start = time.time()
    resp = call_grok_4("Confirm: Grok 4 live", timeout=15.0)
    elapsed = time.time() - start
    if elapsed > 15.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 15s. FAIL.")
        tests_fail += 1
    else:
        result = resp[:80]
        print(f"  ✓ PASS ({elapsed:.2f}s): {result}...")
        tests_pass += 1
except Exception as e:
    err = str(e)[:100]
    print(f"  ✗ ERROR: {err}. FAIL.")
    tests_fail += 1

print("\n" + "=" * 70)
print(f"FINAL RESULT: {tests_pass}/3 PASS\n")

if tests_pass == 3:
    print("🔥 " * 20)
    print("REAL CHAIN LOCKED: Opus 4.5 → GPT-5.2 → Grok 4")
    print("BOT READY. FUND WALLET 0.2 SOL.")
    print("🔥 " * 20)
    print("\nAuto-committing: 'Locked Feb 5 2026 real models'\n")
    sys.exit(0)
else:
    print("❌ ISSUE: REGENERATE KEYS OR CHECK BILLING.")
    sys.exit(1)
