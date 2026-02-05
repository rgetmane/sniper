#!/usr/bin/env python3
"""Full sync verification: Claude + GPT + Grok with 2s timeout each."""

import sys
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import requests

def call_grok(prompt):
    """Minimal Grok HTTP call."""
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-latest",
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
        raise Exception(f"Grok failed: {e}")

tests_pass = 0
tests_fail = 0

# Test 1: Claude "Am I alive?" → expect "Yes"
print("[TEST 1] Claude: Am I alive?")
try:
    start = time.time()
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=os.getenv('ANTHROPIC_API_KEY'))
    resp = llm.invoke([HumanMessage(content="Am I alive? (answer only with Yes or No)")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    elif "yes" in resp.content.lower():
        print(f"  ✓ PASS: Yes ({elapsed:.1f}s)")
        tests_pass += 1
    else:
        print(f"  ✗ UNEXPECTED: {resp.content[:50]} ({elapsed:.1f}s). FAIL.")
        tests_fail += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

# Test 2: GPT "What's 2+2?" → expect "4"
print("[TEST 2] GPT: What's 2+2?")
try:
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv('OPENAI_API_KEY'), temperature=0)
    resp = llm.invoke([HumanMessage(content="What is 2+2? Answer with only the number.")])
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    elif "4" in resp.content:
        print(f"  ✓ PASS: 4 ({elapsed:.1f}s)")
        tests_pass += 1
    else:
        print(f"  ✗ UNEXPECTED: {resp.content[:50]} ({elapsed:.1f}s). FAIL.")
        tests_fail += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

# Test 3: Grok "Are we live?" → expect "Yes"
print("[TEST 3] Grok: Are we live?")
try:
    start = time.time()
    resp = call_grok("Are we live? (answer only with Yes or No)")
    elapsed = time.time() - start
    if elapsed > 2.0:
        print(f"  ✗ TIMEOUT: {elapsed:.1f}s > 2s. FAIL.")
        tests_fail += 1
    elif "yes" in resp.lower():
        print(f"  ✓ PASS: Yes ({elapsed:.1f}s)")
        tests_pass += 1
    else:
        print(f"  ✗ UNEXPECTED: {resp[:50]} ({elapsed:.1f}s). FAIL.")
        tests_fail += 1
except Exception as e:
    print(f"  ✗ ERROR: {str(e)[:100]}. FAIL.")
    tests_fail += 1

print(f"\n=== RESULT: {tests_pass}/3 PASS ===")
if tests_pass == 3:
    print("\n" + "🔥 " * 15)
    print("FULL CHAIN LIVE. MODELS ONLINE. BOT READY.")
    print("🔥 " * 15)
    sys.exit(0)
else:
    print("\n❌ SYNC DEAD. CHECK KEYS.")
    sys.exit(1)
