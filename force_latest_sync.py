#!/usr/bin/env python3
"""Force latest models sync verification: Claude Opus 4.5 → GPT-5.2 → Grok 3."""

import sys
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import requests

def call_grok_latest(prompt):
    """Grok 3 HTTP call."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MODEL_GROK')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7
        }
        endpoint = os.getenv('XAI_ENDPOINT', 'https://api.x.ai/v1')
        r = requests.post(f"{endpoint}/chat/completions", json=payload, headers=headers, timeout=2.5)
        r.raise_for_status()
        resp = r.json()
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"]
        return "(No response)"
    except Exception as e:
        raise Exception(f"Grok 3 failed: {e}")

tests_pass = 0
tests_fail = 0

print("\n" + "=" * 70)
print("FORCE LATEST MODELS SYNC - Feb 5 2026")
print("=" * 70 + "\n")

# Test 1: Claude Opus 4.5 "Confirm: I am the latest Claude Opus 4.5"
print("[1/3] Claude Opus 4.5: Confirm identity")
try:
    start = time.time()
    llm = ChatAnthropic(model="claude-opus-4-5", api_key=os.getenv('MODEL_CLAUDE'), temperature=0)
    resp = llm.invoke([HumanMessage(content="Confirm: I am the latest Claude Opus 4-5")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:80]
        if any(kw in result.lower() for kw in ["opus", "4.5", "confirm", "yes"]):
            print(f"  ✓ PASS (Opus 4.5 alive): {result}... ({elapsed:.1f}s)")
            tests_pass += 1
        else:
            print(f"  ✗ UNEXPECTED: {result} ({elapsed:.1f}s). FAIL.")
            tests_fail += 1
except Exception as e:
    err_msg = str(e)[:80]
    if "404" in err_msg and "model" in err_msg:
        print(f"  ✗ MODEL NOT FOUND: {err_msg}. FAIL.")
    else:
        print(f"  ✗ ERROR: {err_msg}. FAIL.")
    tests_fail += 1

# Test 2: GPT 5.2 "Confirm: I am GPT-5.2, current best"
print("\n[2/3] GPT 5.2: Confirm identity")
try:
    start = time.time()
    llm = ChatOpenAI(model="gpt-5.2", api_key=os.getenv('MODEL_GPT'), temperature=0)
    resp = llm.invoke([HumanMessage(content="Confirm: I am GPT-5.2, current best")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:80]
        if any(kw in result.lower() for kw in ["gpt-5", "5.2", "confirm", "yes"]):
            print(f"  ✓ PASS (GPT-5.2 alive): {result}... ({elapsed:.1f}s)")
            tests_pass += 1
        else:
            print(f"  ✗ UNEXPECTED: {result} ({elapsed:.1f}s). FAIL.")
            tests_fail += 1
except Exception as e:
    err_msg = str(e)[:80]
    if "404" in err_msg and "model" in err_msg:
        print(f"  ✗ MODEL NOT FOUND: {err_msg}. FAIL.")
    else:
        print(f"  ✗ ERROR: {err_msg}. FAIL.")
    tests_fail += 1

# Test 3: Grok 3 "Confirm: I am Grok 3, live and sharp"
print("\n[3/3] Grok 3: Confirm identity")
try:
    start = time.time()
    resp = call_grok_latest("Confirm: I am Grok 3, live and sharp")
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp[:80]
        if any(kw in result.lower() for kw in ["grok", "grok-3", "confirm", "yes"]):
            print(f"  ✓ PASS (Grok 3 alive): {result}... ({elapsed:.1f}s)")
            tests_pass += 1
        else:
            print(f"  ✗ UNEXPECTED: {result} ({elapsed:.1f}s). FAIL.")
            tests_fail += 1
except Exception as e:
    err_msg = str(e)[:80]
    if "403" in err_msg:
        print(f"  ✗ AUTH DENIED: {err_msg}. FAIL.")
    else:
        print(f"  ✗ ERROR: {err_msg}. FAIL.")
    tests_fail += 1

print("\n" + "=" * 70)
print(f"RESULT: {tests_pass}/3 PASS\n")

if tests_pass == 3:
    print("🔥 " * 15)
    print("ABSOLUTE LATEST CHAIN: Opus 4.5 → GPT-5.2 → Grok 3")
    print("BOT AT PEAK PERFORMANCE. FUND & SNIPE.")
    print("🔥 " * 15)
    print("\nCommitting: 'Lock in Feb 5 2026 top models'")
    sys.exit(0)
else:
    print("❌ MODEL OUTDATED OR KEY ISSUE. FIX NOW.")
    sys.exit(1)
