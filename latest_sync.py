#!/usr/bin/env python3
"""Force latest models and sync: Claude Opus 4.5 → GPT-5.2 → Grok 4."""

import sys
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import requests

def call_grok_4(prompt):
    """Grok 4 HTTP call with 2s timeout."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('MODEL_CLAUDE_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-4",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7
        }
        endpoint = os.getenv('XAI_ENDPOINT', 'https://api.x.ai/v1/chat/completions')
        r = requests.post(endpoint, json=payload, headers=headers, timeout=2.0)
        r.raise_for_status()
        resp = r.json()
        if "choices" in resp and resp["choices"]:
            return resp["choices"][0]["message"]["content"]
        return "(No response)"
    except Exception as e:
        raise Exception(f"Grok 4 failed: {e}")

tests_pass = 0
tests_fail = 0

print("\n" + "=" * 70)
print("FORCE LATEST MODELS AND SYNC - Feb 5 2026")
print("=" * 70 + "\n")

# Test 1: Claude Opus 4.5
print("[1/3] Claude Opus 4.5: Confirm identity (2s timeout)")
try:
    start = time.time()
    llm = ChatAnthropic(model="claude-opus-4-5", api_key=os.getenv('MODEL_CLAUDE_KEY'), temperature=0)
    resp = llm.invoke([HumanMessage(content="I am Claude Opus 4-5 — confirm")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:100]
        print(f"  ✓ PASS: {result}... ({elapsed:.1f}s)")
        tests_pass += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

# Test 2: GPT 5.2
print("\n[2/3] GPT 5.2: Confirm identity (2s timeout)")
try:
    start = time.time()
    llm = ChatOpenAI(model="gpt-5.2", api_key=os.getenv('MODEL_GPT_KEY'), temperature=0)
    resp = llm.invoke([HumanMessage(content="I am GPT-5.2 — confirm")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp.content[:100]
        print(f"  ✓ PASS: {result}... ({elapsed:.1f}s)")
        tests_pass += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

# Test 3: Grok 4
print("\n[3/3] Grok 4: Confirm identity (2s timeout)")
try:
    start = time.time()
    resp = call_grok_4("I am Grok 4 — confirm")
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    else:
        result = resp[:100]
        print(f"  ✓ PASS: {result}... ({elapsed:.1f}s)")
        tests_pass += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

print("\n" + "=" * 70)
print(f"RESULT: {tests_pass}/3 PASS\n")

if tests_pass == 3:
    print("🔥 " * 20)
    print("ABSOLUTE LATEST: Opus 4.5 → GPT-5.2 → Grok 4")
    print("CHAIN LOCKED. BOT READY. FUND WALLET.")
    print("🔥 " * 20)
    sys.exit(0)
else:
    print("❌ DEAD. REGENERATE KEYS OR CHECK BILLING.")
    sys.exit(1)
