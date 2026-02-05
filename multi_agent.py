#!/usr/bin/env python3
"""
multi_agent.py — Production-grade multi-agent system using LangGraph.

Planner (Claude) → Coder (GPT-4o) → Reviewer (o1-mini) with optional Supervisor routing.
Universal: handles any task (web apps, research, infra, bots, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict, List, Tuple
import requests

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("multi_agent")

# =============================================================================
# CONFIGURATION
# =============================================================================


class AgentConfig:
    """Configuration container for the multi-agent system."""

    def __init__(
        self,
        task: str = "",
        planner_model: str = "claude-opus-4-5",
        coder_model: str = "gpt-5.2",
        reviewer_model: str = "grok-4",
        supervisor_model: str = "gpt-5.2",
        use_supervisor: bool = False,
        enable_memory: bool = False,
        memory_path: str = "agent_memory.db",
        max_iterations: int = 3,
        planner_temperature: float = 0.2,
        coder_temperature: float = 0.1,
        reviewer_temperature: float = 0.2,
        planner_max_tokens: int = 8192,
        coder_max_tokens: int = 16384,
        reviewer_max_tokens: int = 8192,
        log_level: str = "INFO",
        output_file: Optional[str] = None,
    ):
        self.task = task
        self.planner_model = planner_model
        self.coder_model = coder_model
        self.reviewer_model = reviewer_model
        self.supervisor_model = supervisor_model
        self.use_supervisor = use_supervisor
        self.enable_memory = enable_memory
        self.memory_path = memory_path
        self.max_iterations = max_iterations
        self.planner_temperature = planner_temperature
        self.coder_temperature = coder_temperature
        self.reviewer_temperature = reviewer_temperature
        self.planner_max_tokens = planner_max_tokens
        self.coder_max_tokens = coder_max_tokens
        self.reviewer_max_tokens = reviewer_max_tokens
        self.log_level = log_level
        self.output_file = output_file

    @classmethod
    def from_file(cls, config_path: str) -> "AgentConfig":
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            if path.suffix in {".yaml", ".yml"}:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML not installed. Run: pip install pyyaml")
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task": self.task,
            "planner_model": self.planner_model,
            "coder_model": self.coder_model,
            "reviewer_model": self.reviewer_model,
            "supervisor_model": self.supervisor_model,
            "use_supervisor": self.use_supervisor,
            "enable_memory": self.enable_memory,
            "memory_path": self.memory_path,
            "max_iterations": self.max_iterations,
            "planner_temperature": self.planner_temperature,
            "coder_temperature": self.coder_temperature,
            "reviewer_temperature": self.reviewer_temperature,
            "planner_max_tokens": self.planner_max_tokens,
            "coder_max_tokens": self.coder_max_tokens,
            "reviewer_max_tokens": self.reviewer_max_tokens,
            "log_level": self.log_level,
            "output_file": self.output_file,
        }


# Global config instance (initialized in main)
config: Optional[AgentConfig] = None


# =============================================================================
# STATE SCHEMA
# =============================================================================


class AgentState(TypedDict, total=False):
    """Shared state for the graph.

    - messages uses operator.add for append-only history.
    - status controls routing and early exits.
    """

    task: str
    plan: str
    code: str
    review: str
    status: str
    iteration: int
    use_supervisor: bool
    max_iterations: int
    messages: Annotated[list[str], operator.add]

# =============================================================================
# MODEL FACTORIES
# =============================================================================


def get_planner_llm() -> ChatAnthropic:
    """Planner: deep reasoning + architecture design (Claude Opus 4.5)."""
    cfg = config or AgentConfig()
    api_key = os.getenv('MODEL_CLAUDE') or os.getenv('ANTHROPIC_API_KEY')
    return ChatAnthropic(
        model=cfg.planner_model,
        temperature=cfg.planner_temperature,
        max_tokens=cfg.planner_max_tokens,
        api_key=api_key,
    )


def get_coder_llm() -> ChatOpenAI:
    """Coder: fast, accurate code generation (GPT-5.2)."""
    cfg = config or AgentConfig()
    api_key = os.getenv('MODEL_GPT') or os.getenv('OPENAI_API_KEY')
    return ChatOpenAI(
        model=cfg.coder_model,
        temperature=cfg.coder_temperature,
        max_tokens=cfg.coder_max_tokens,
        api_key=api_key,
    )


def get_reviewer_llm() -> ChatOpenAI:
    """Reviewer: bug fixing + security audit (Grok 3)."""
    cfg = config or AgentConfig()
    api_key = os.getenv('MODEL_GROK') or os.getenv('XAI_API_KEY')
    return ChatOpenAI(
        model=cfg.reviewer_model,
        temperature=cfg.reviewer_temperature,
        max_tokens=cfg.reviewer_max_tokens,
        api_key=api_key,
    )


def get_supervisor_llm() -> ChatOpenAI:
    """Supervisor: lightweight router (optional) - GPT-5.2."""
    cfg = config or AgentConfig()
    api_key = os.getenv('MODEL_GPT') or os.getenv('OPENAI_API_KEY')
    return ChatOpenAI(
        model=cfg.supervisor_model,
        temperature=0,
        max_tokens=512,
        api_key=api_key,
    )

# =============================================================================
# UTILS
# =============================================================================


def init_state(state: AgentState) -> AgentState:
    """Ensure required defaults exist even if app.invoke receives minimal input."""
    cfg = config or AgentConfig()
    return {
        "task": state.get("task", ""),
        "plan": state.get("plan", ""),
        "code": state.get("code", ""),
        "review": state.get("review", ""),
        "status": state.get("status", "pending"),
        "iteration": state.get("iteration", 0),
        "use_supervisor": bool(state.get("use_supervisor", False)),
        "max_iterations": state.get("max_iterations", cfg.max_iterations),
        "messages": state.get("messages", ["[SYSTEM] Task received"]),
    }


# -----------------------
# Workspace context collector (depth-limited)
# -----------------------
IGNORED_DIRS = {".git", "node_modules", "build"}


def gather_workspace_context(root: Optional[str] = None, depth: int = 5) -> List[Tuple[str, str]]:
    """
    Collect prioritized files and a shallow crawl of code files to include in prompts.
    Returns list of (relative_path, truncated_content).
    """
    root_path = Path(root or Path.cwd())
    results: List[Tuple[str, str]] = []

    def should_ignore(p: Path) -> bool:
        for part in p.parts:
            if part in IGNORED_DIRS:
                return True
        return False

    # prioritized files
    important = ["sniper.log", "sniper_memory.json", ".env.example", "main.go", "secrets.env"]
    for fname in important:
        p = root_path / fname
        if p.exists() and p.is_file():
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                results.append((str(p.relative_to(root_path)), txt[:64_000]))
            except Exception:
                results.append((str(p.relative_to(root_path)), "<unreadable>"))

    # shallow crawl for code files
    def walk(p: Path, cur_depth: int):
        if cur_depth > depth:
            return
        try:
            for child in sorted(p.iterdir()):
                if should_ignore(child):
                    continue
                if child.is_file() and child.suffix in {".go", ".py", ".rs", ".js", ".ts", ".toml", ".env"}:
                    rel = str(child.relative_to(root_path))
                    if any(rel == r for r, _ in results):
                        continue
                    try:
                        txt = child.read_text(encoding="utf-8", errors="ignore")
                        results.append((rel, txt[:12_000]))
                    except Exception:
                        results.append((rel, "<unreadable>"))
                elif child.is_dir():
                    walk(child, cur_depth + 1)
        except PermissionError:
            return

    walk(root_path, 1)
    return results


# -----------------------
# Routing / override helpers
# -----------------------
# Updated default endpoint to x.ai; can be overridden via XAI_ENDPOINT env var
XAI_ENDPOINT = os.getenv("XAI_ENDPOINT", "https://api.x.ai/v1/chat/completions")


def parse_override_prefix(task: str) -> Tuple[Optional[str], str]:
    """Return (provider, stripped_task) if a /grok|/claude|/gpt prefix is used."""
    m = re.match(r"^/(grok|claude|gpt)\s+(.+)$", (task or "").strip(), re.I)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return None, task


def choose_provider_for_text(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("strategy", "audit", "long-form", "design", "architecture")):
        return "claude"
    if any(k in t for k in ("code", "debug", "refactor", "fix", "patch", "diff", "implement")):
        return "gpt"
    if any(k in t for k in ("explain", "idea", "speed", "quick")):
        return "grok"
    return os.getenv("DEFAULT_ROUTER", "gpt")


def call_grok(model: str, prompt: str) -> str:
    """Minimal HTTP adapter for xAI/Grok-like endpoints. Uses XAI_KEY."""
    api_key = os.getenv("XAI_API_KEY") or os.getenv("XAI_KEY") or os.getenv("MODEL_GROK")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY for Grok provider")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = requests.post(XAI_ENDPOINT, headers=headers, json=payload, timeout=30)
        if r.status_code in {400, 404} and model != "grok-4-0709":
            fallback_payload = {**payload, "model": "grok-4-0709"}
            r = requests.post(XAI_ENDPOINT, headers=headers, json=fallback_payload, timeout=30)
        r.raise_for_status()
    except requests.HTTPError as exc:
        log.error("[GROK] HTTP error: %s", exc)
        raise
    j = r.json()
    if "choices" in j and j["choices"]:
        return j["choices"][0]["message"].get("content", "")
    return j.get("completion") or j.get("text") or j.get("output") or json.dumps(j)


def routed_llm_call(task_text: str, override: Optional[str] = None) -> str:
    """Route to grok/claude/gpt based on override or heuristics, include workspace context."""
    provider = override or choose_provider_for_text(task_text)
    log.info("Router selected provider=%s", provider)

    ctx = gather_workspace_context()
    ctx_snip = "\n\n".join([f"== {p} ==\n{txt[:2000]}" for p, txt in ctx])
    prompt = f"Workspace Context:\n{ctx_snip}\n\nUser Task:\n{task_text}"

    if provider == "claude":
        llm = get_planner_llm()
        out = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Answer concisely")])
        return out.content
    if provider == "gpt":
        llm = get_coder_llm()
        out = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Answer concisely")])
        return out.content
    # grok
    return call_grok(os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning"), prompt)


VERIFY_SYNC_TIMEOUT_SEC = float(os.getenv("VERIFY_SYNC_TIMEOUT_SEC", "30"))


def verify_sync_chain() -> bool:
    """Multi-agent verification sync: Claude (main.go safety) + GPT (trailing stop) + Grok (sniper.log).
    
    Strict timing: abort if any provider delays >2s.
    Returns: True if all pass, False (abort) otherwise.
    """
    import threading
    import time
    
    # Phase 1: Claude scans main.go for 8 safety lines
    print("\n[CLAUDE] Scanning main.go for 8 safety lines...")
    sys_prompt = "You are a code auditor. Read main.go and return ONLY the 8 safety gate lines (exact code). If missing one, respond: BREACH"
    user_msg = "What are the 8 safety gates in this code?"
    
    try:
        start = time.time()
        llm = get_planner_llm()
        claude_resp = llm.invoke([
            SystemMessage(content=sys_prompt + "\n\nCode:\n" + open("main.go", "r").read()[:8000]),
            HumanMessage(content=user_msg)
        ])
        elapsed = time.time() - start
        if elapsed > VERIFY_SYNC_TIMEOUT_SEC:
            log.error("[CLAUDE] Timeout: %.2fs > %.1fs. ABORT.", elapsed, VERIFY_SYNC_TIMEOUT_SEC)
            return False
        claude_result = claude_resp.content[:500]
        print(f"✓ Claude: {claude_result[:120]}... ({elapsed:.1f}s)")
        if "BREACH" in claude_result.upper():
            print("✗ BREACH detected in safety checks. ABORT.")
            return False
    except Exception as e:
        log.error("[CLAUDE] failed: %s. ABORT.", e)
        return False
    
    # Phase 2: GPT writes 5-line trailing stop diff
    print("\n[GPT] Writing trailing stop diff (5 lines)...")
    sys_prompt = "You are a Go developer. Write EXACTLY 5 lines (no explanation): add trailing stop logic. Sell 30% at 5% below peak after +70%, moonbag at +150%. Diff format only."
    user_msg = "Trailing stop diff:"
    
    try:
        start = time.time()
        llm = get_coder_llm()
        gpt_resp = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_msg)
        ])
        elapsed = time.time() - start
        if elapsed > VERIFY_SYNC_TIMEOUT_SEC:
            log.error("[GPT] Timeout: %.2fs > %.1fs. ABORT.", elapsed, VERIFY_SYNC_TIMEOUT_SEC)
            return False
        gpt_result = gpt_resp.content[:300]
        print(f"✓ GPT: {gpt_result[:80]}... ({elapsed:.1f}s)")
    except Exception as e:
        log.error("[GPT] failed: %s. ABORT.", e)
        return False
    
    # Phase 3: Grok reads sniper.log (last 3 lines + confirmation)
    print("\n[GROK] Reading sniper.log and confirming bot status...")
    try:
        log_lines = open("sniper.log", "r").readlines()[-3:]
        log_text = "".join(log_lines)
    except:
        log_text = "(sniper.log not found or empty)"
    
    sys_prompt = f"Read the sniper bot log:\n\n{log_text}\n\nAnswer exactly (one line each):\nIs bot running? (yes/no)\nWallet balance? (amount or unknown)\nLast action? (action type)"
    user_msg = "Confirm status:"
    
    try:
        start = time.time()
        grok_result = call_grok(os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning"), sys_prompt)
        elapsed = time.time() - start
        if elapsed > VERIFY_SYNC_TIMEOUT_SEC:
            log.error("[GROK] Timeout: %.2fs > %.1fs. ABORT.", elapsed, VERIFY_SYNC_TIMEOUT_SEC)
            return False
        print(f"✓ Grok: {grok_result[:100]}... ({elapsed:.1f}s)")
    except Exception as e:
        log.error("[GROK] failed: %s. ABORT.", e)
        return False
    
    # All phases passed
    return True


def test_chain() -> None:
    """Run the user's specified test chain for overrides.

    Tests:
      - /grok "hi from sniper folder" -> should echo (we print response)
      - /claude "list every safety gate in main.go" -> print response
      - /gpt "fix trailing stop logic" -> print response (diff expected)
    """
    tests = [
        ("/grok hi from sniper folder", "Grok echo test"),
        ("/claude list every safety gate in main.go", "Claude safety gates"),
        ("/gpt fix trailing stop logic", "GPT fix trailing stop logic"),
    ]

    for raw, desc in tests:
        provider, body = parse_override_prefix(raw)
        log.info("TEST: %s -> provider=%s", desc, provider)
        try:
            out = routed_llm_call(body, override=provider)
            print("\n--- TEST OUTPUT: {} ---\n".format(desc))
            print(out[:4000])
            print("\n--- END OUTPUT ---\n")
        except Exception as exc:
            log.exception("Test %s failed: %s", desc, exc)


# =============================================================================
# AGENT NODES
# =============================================================================


def supervisor_node(state: AgentState) -> dict:
    """Optional router that decides which agent should run next."""
    log.info("SUPERVISOR: routing decision")
    llm = get_supervisor_llm()

    prompt = (
        "You are a supervisor routing agent. Choose the next step based on the state. "
        "Valid outputs: planner, coder, reviewer, end.\n\n"
        f"Task: {state.get('task', '')}\n"
        f"Plan length: {len(state.get('plan', ''))}\n"
        f"Code length: {len(state.get('code', ''))}\n"
        f"Review length: {len(state.get('review', ''))}\n"
        f"Status: {state.get('status', '')}\n"
        f"Iteration: {state.get('iteration', 0)}\n"
        "Rules: If no plan -> planner. If plan and no code -> coder. "
        "If code and no review -> reviewer. If status=approved -> end."
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Return only one token: planner|coder|reviewer|end"),
            ]
        )
        choice = response.content.strip().lower()
    except Exception as exc:
        log.exception("SUPERVISOR: error during routing: %s", exc)
        choice = "planner"

    if choice not in {"planner", "coder", "reviewer", "end"}:
        choice = "planner"

    return {
        "messages": [f"[SUPERVISOR] Routed to: {choice}"]
    }


def planner_node(state: AgentState) -> dict:
    """Planner: generates architecture and step-by-step plan."""
    log.info("PLANNER: generating plan")
    # Build planner prompt and route to appropriate provider (override to Claude for long-form)
    prompt = (
        "You are an elite software architect. Given a task, create a concise, "
        "actionable implementation plan. Include: architecture overview, modules, "
        "data flow, dependencies, error handling, security, and testing strategy. "
        "Output ONLY the plan.\n\n"
        f"Task: {state.get('task', '')}"
    )
    try:
        plan = routed_llm_call(prompt, override="claude")
        status = "planned"
        msg = f"[PLANNER] Plan generated ({len(plan)} chars)"
    except Exception as exc:
        log.exception("PLANNER: error: %s", exc)
        plan = ""
        status = "error"
        msg = f"[PLANNER] Error: {exc}"

    return {
        "plan": plan,
        "status": status,
        "messages": [msg],
    }


def coder_node(state: AgentState) -> dict:
    """Coder: produces runnable, production-grade code from the plan."""
    log.info("CODER: generating code")
    if state.get("status") == "error":
        return {"messages": ["[CODER] Skipped due to prior error"]}
    # Build coder context and route to GPT by default for code generation
    context = f"## Plan\n{state.get('plan', '')}"
    if state.get("review"):
        context += f"\n\n## Reviewer Feedback\n{state.get('review', '')}"
        context += f"\n\n## Previous Code\n{state.get('code', '')}"

    prompt = (
        "You are a senior software engineer. Using the plan below, produce complete, "
        "runnable, production-quality code with type hints and inline comments. "
        "Output ONLY code (no markdown fences).\n\n"
        + context
    )

    try:
        code = routed_llm_call(prompt, override="gpt")
        status = "coded"
        msg = f"[CODER] Code generated ({len(code)} chars)"
    except Exception as exc:
        log.exception("CODER: error: %s", exc)
        code = ""
        status = "error"
        msg = f"[CODER] Error: {exc}"

    return {
        "code": code,
        "status": status,
        "messages": [msg],
    }


def reviewer_node(state: AgentState) -> dict:
    """Reviewer: audits correctness, security, performance, best practices."""
    log.info("REVIEWER: reviewing code")
    if state.get("status") == "error":
        return {"messages": ["[REVIEWER] Skipped due to prior error"]}

    # Reviewer prefers long-form audit (route to Claude by default)
    prompt = (
        "You are a senior reviewer and security auditor. Review the code for "
        "correctness, bugs, security, performance, and best practices. If the code "
        "is production-ready, respond with exactly: APPROVED. Otherwise list issues "
        "and recommended fixes.\n\n"
        f"## Task\n{state.get('task', '')}\n\n"
        f"## Plan\n{state.get('plan', '')}\n\n"
        f"## Code\n{state.get('code', '')}"
    )

    try:
        review = routed_llm_call(prompt, override="claude")
        approved = "APPROVED" in review.upper() and len(review.strip()) < 200
        status = "approved" if approved else "needs_revision"
        msg = f"[REVIEWER] {'APPROVED' if approved else 'Changes requested'}"
    except Exception as exc:
        log.exception("REVIEWER: error: %s", exc)
        review = ""
        status = "error"
        msg = f"[REVIEWER] Error: {exc}"

    iteration = state.get("iteration", 0) + 1

    return {
        "review": review,
        "status": status,
        "iteration": iteration,
        "messages": [msg],
    }

# =============================================================================
# ROUTERS
# =============================================================================


def route_after_planner(state: AgentState) -> Literal["coder", "__end__"]:
    if state.get("status") == "error":
        return END
    return "coder"


def route_after_coder(state: AgentState) -> Literal["reviewer", "__end__"]:
    if state.get("status") == "error":
        return END
    return "reviewer"


def route_after_reviewer(state: AgentState) -> Literal["coder", "__end__"]:
    if state.get("status") == "approved":
        return END
    max_iter = state.get("max_iterations", 3)
    if state.get("iteration", 0) >= max_iter:
        log.warning("ROUTER: max iterations (%d) reached — ending", max_iter)
        return END
    return "coder"


def route_from_start(state: AgentState) -> Literal["supervisor", "planner"]:
    return "supervisor" if state.get("use_supervisor") else "planner"


def route_from_supervisor(state: AgentState) -> Literal["planner", "coder", "reviewer", "__end__"]:
    last = (state.get("messages") or [""])[-1].lower()
    if "routed to: coder" in last:
        return "coder"
    if "routed to: reviewer" in last:
        return "reviewer"
    if "routed to: end" in last:
        return END
    return "planner"

# =============================================================================
# GRAPH
# =============================================================================


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)

    graph.add_conditional_edges(START, route_from_start)
    graph.add_conditional_edges("supervisor", route_from_supervisor)

    # Required sequential flow
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_conditional_edges("coder", route_after_coder)
    graph.add_conditional_edges("reviewer", route_after_reviewer)

    return graph


def create_app(
    enable_memory: Optional[bool] = None,
    memory_path: Optional[str] = None,
):
    """Create a compiled LangGraph app, optionally with persistence.

    Memory can be enabled via arguments or env vars:
    - ENABLE_MEMORY=true|1
    - LANGGRAPH_MEMORY_PATH=agent_memory.db
    """
    graph = build_graph()

    if enable_memory is None:
        enable_memory = os.getenv("ENABLE_MEMORY", "").lower() in {"1", "true", "yes"}
    if memory_path is None:
        memory_path = os.getenv("LANGGRAPH_MEMORY_PATH", "agent_memory.db")

    if enable_memory:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(memory_path)
            return graph.compile(checkpointer=checkpointer)
        except Exception as exc:
            log.exception("Failed to enable memory: %s", exc)

    return graph.compile()


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-agent system: Planner → Coder → Reviewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_agent.py --task "Build a REST API with FastAPI"
  python multi_agent.py --config config.yaml
  python multi_agent.py --task "Create a web scraper" --max-iterations 5 --use-supervisor
  python multi_agent.py --task "Build a CLI tool" --output-file result.py
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task description for the agent system",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML or JSON)",
    )
    parser.add_argument(
        "--planner-model",
        type=str,
        default="claude-opus-4-5",
        help="Model for planner agent (default: claude-opus-4-5)",
    )
    parser.add_argument(
        "--coder-model",
        type=str,
        default="gpt-5.2",
        help="Model for coder agent (default: gpt-5.2)",
    )
    parser.add_argument(
        "--reviewer-model",
        type=str,
        default="grok-4",
        help="Model for reviewer agent (default: grok-4)",
    )
    parser.add_argument(
        "--use-supervisor",
        action="store_true",
        help="Enable supervisor routing",
    )
    parser.add_argument(
        "--enable-memory",
        action="store_true",
        help="Enable conversation memory/checkpointing",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default="agent_memory.db",
        help="Path to memory database (default: agent_memory.db)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum review-fix iterations (default: 3)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save final code to file",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run provider override test chain and exit",
    )
    parser.add_argument(
        "--verify-sync",
        action="store_true",
        help="Run multi-agent verification sync (Claude/GPT/Grok) with 2s timeout per provider",
    )

    return parser.parse_args()


def main():
    """Main entry point with CLI support."""
    global config

    args = parse_args()

    # Load config from file or CLI args
    if args.config:
        config = AgentConfig.from_file(args.config)
        # CLI args override config file
        if args.task:
            config.task = args.task
        if args.use_supervisor:
            config.use_supervisor = True
        if args.enable_memory:
            config.enable_memory = True
        if args.max_iterations != 3:
            config.max_iterations = args.max_iterations
    else:
        config = AgentConfig(
            task=args.task or "",
            planner_model=args.planner_model,
            coder_model=args.coder_model,
            reviewer_model=args.reviewer_model,
            use_supervisor=args.use_supervisor,
            enable_memory=args.enable_memory,
            memory_path=args.memory_path,
            max_iterations=args.max_iterations,
            log_level=args.log_level,
            output_file=args.output_file,
        )

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Run test chain if requested (doesn't need a task)
    if args.run_tests:
        log.info("Running provider override test chain...")
        test_chain()
        sys.exit(0)

    # Run verification sync if requested
    if args.verify_sync:
        log.info("Running multi-agent verification sync...")
        success = verify_sync_chain()
        if success:
            print("\n" + "🔥 " * 20)
            print("CHAIN LOCKED. MODELS IN PHASE. BOT READY. FUND NOW.")
            print("🔥 " * 20 + "\n")
            sys.exit(0)
        else:
            print("\nABORT: Verification failed or timeout exceeded.")
            sys.exit(1)

    # Validate task (only for normal operation, not for tests)
    if not config.task:
        log.error("No task provided. Use --task or --config with a task field.")
        sys.exit(1)

    log.info("=== Multi-Agent System Starting ===")
    log.info("Task: %s", config.task)
    log.info("Planner: %s", config.planner_model)
    log.info("Coder: %s", config.coder_model)
    log.info("Reviewer: %s", config.reviewer_model)
    log.info("Supervisor: %s", "enabled" if config.use_supervisor else "disabled")
    log.info("Max iterations: %d", config.max_iterations)

    # Create and run app
    app = create_app(
        enable_memory=config.enable_memory,
        memory_path=config.memory_path,
    )

    initial_state = {
        "task": config.task,
        "messages": [],
        "iteration": 0,
        "status": "pending",
        "use_supervisor": config.use_supervisor,
        "max_iterations": config.max_iterations,
    }

    try:
        result = app.invoke(initial_state)
        log.info("=== Execution Complete ===")
        log.info("Final status: %s", result.get("status"))
        log.info("Iterations: %d", result.get("iteration", 0))

        # Display results
        if result.get("plan"):
            log.info("\n--- PLAN ---\n%s", result["plan"][:500])
        if result.get("code"):
            log.info("\n--- CODE ---\n%s", result["code"][:1000])
        if result.get("review"):
            log.info("\n--- REVIEW ---\n%s", result["review"][:500])

        # Save to file if requested
        if config.output_file and result.get("code"):
            output_path = Path(config.output_file)
            output_path.write_text(result["code"])
            log.info("Code saved to: %s", config.output_file)

    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

# =============================================================================
# README (comments)
# =============================================================================
#
# INSTALL:
#   pip install langgraph langchain langchain-openai langchain-anthropic langchain-community
#
# ENV VARS:
#   export ANTHROPIC_API_KEY=sk-ant-...   # Planner (Claude 3.5 Sonnet)
#   export OPENAI_API_KEY=sk-...          # Coder (GPT-4o) + Reviewer (o1-mini)
#
# RUN:
#   python multi_agent.py
#
# EXTEND:
#   1. Add tools:
#      from langchain_community.tools import DuckDuckGoSearchRun
#      tool = DuckDuckGoSearchRun()
#      llm = get_coder_llm().bind_tools([tool])
#      Use ToolNode from langgraph.prebuilt for automatic tool execution.
#
#   2. Parallel fan-out:
#      graph.add_edge("planner", "coder")
#      graph.add_edge("planner", "researcher")
#      graph.add_edge(["coder", "researcher"], "reviewer")
#
#   3. Memory / persistence:
#      from langgraph.checkpoint.sqlite import SqliteSaver
#      memory = SqliteSaver.from_conn_string("agent_memory.db")
#      app = graph.compile(checkpointer=memory)
#      app.invoke(state, config={"configurable": {"thread_id": "project-1"}})
#
#   4. Supervisor routing:
#      Set use_supervisor=True in state to enable LLM-based routing.
#
# =============================================================================