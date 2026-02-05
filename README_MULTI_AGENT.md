# Multi-Agent System

Production-grade multi-agent system using LangGraph with **Planner → Coder → Reviewer** workflow.

## Features

✅ **Three specialized agents**: Claude (Planner), GPT-4o (Coder), o1-mini (Reviewer)  
✅ **CLI + Config file support**: Run via command-line arguments or YAML/JSON config  
✅ **Configurable iteration loops**: Set max review-fix cycles  
✅ **Optional supervisor routing**: LLM-based dynamic agent routing  
✅ **Memory/persistence**: SQLite checkpointing for conversation history  
✅ **Flexible output**: Save generated code to file  
✅ **Comprehensive logging**: Configurable log levels (DEBUG/INFO/WARNING/ERROR)

---

## Installation

```bash
pip install langgraph langchain langchain-openai langchain-anthropic langchain-community
```

### Optional: YAML support
```bash
pip install pyyaml
```

---

## Quick Start

### 1. Set API Keys
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### 2. Run with CLI
```bash
# Simple task
python multi_agent.py --task "Build a REST API with FastAPI"

# Advanced options
python multi_agent.py \
  --task "Create a web scraper" \
  --max-iterations 5 \
  --use-supervisor \
  --output-file scraper.py \
  --log-level DEBUG
```

### 3. Run with Config File
```bash
# YAML config
python multi_agent.py --config agent_config.yaml

# JSON config
python multi_agent.py --config agent_config.json

# Override config with CLI args
python multi_agent.py --config agent_config.yaml --task "Different task"
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | string | - | Task description (required if no config) |
| `--config` | string | - | Path to YAML/JSON config file |
| `--planner-model` | string | `claude-3-5-sonnet-20241022` | Model for planner agent |
| `--coder-model` | string | `gpt-4o` | Model for coder agent |
| `--reviewer-model` | string | `o1-mini` | Model for reviewer agent |
| `--use-supervisor` | flag | `false` | Enable supervisor routing |
| `--enable-memory` | flag | `false` | Enable conversation memory |
| `--memory-path` | string | `agent_memory.db` | Path to SQLite memory DB |
| `--max-iterations` | int | `3` | Maximum review-fix loops |
| `--log-level` | string | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `--output-file` | string | - | Save generated code to file |

---

## Configuration File Format

### YAML Example (`agent_config.yaml`)
```yaml
task: "Build a REST API with FastAPI"
planner_model: "claude-3-5-sonnet-20241022"
coder_model: "gpt-4o"
reviewer_model: "o1-mini"
use_supervisor: false
max_iterations: 3
enable_memory: false
memory_path: "agent_memory.db"
planner_temperature: 0.2
coder_temperature: 0.1
reviewer_temperature: 0.2
planner_max_tokens: 8192
coder_max_tokens: 16384
reviewer_max_tokens: 8192
log_level: "INFO"
output_file: "generated_code.py"
```

### JSON Example (`agent_config.json`)
```json
{
  "task": "Create a web scraper",
  "planner_model": "claude-3-5-sonnet-20241022",
  "coder_model": "gpt-4o",
  "reviewer_model": "o1-mini",
  "max_iterations": 3,
  "output_file": "scraper.py"
}
```

---

## Usage Examples

### Basic Task
```bash
python multi_agent.py --task "Build a CLI calculator in Python"
```

### With Custom Models
```bash
python multi_agent.py \
  --task "Create a data analysis script" \
  --coder-model "gpt-4-turbo" \
  --reviewer-model "o1"
```

### Enable Supervisor Routing
```bash
python multi_agent.py \
  --task "Build a microservice" \
  --use-supervisor
```

### With Memory (Conversation Persistence)
```bash
python multi_agent.py \
  --task "Build a chatbot" \
  --enable-memory \
  --memory-path my_project.db
```

### High Iteration Count
```bash
python multi_agent.py \
  --task "Complex algorithm implementation" \
  --max-iterations 10
```

### Save Output to File
```bash
python multi_agent.py \
  --task "Build a web server" \
  --output-file server.py
```

### Debug Mode
```bash
python multi_agent.py \
  --task "Build a parser" \
  --log-level DEBUG
```

---

## How It Works

### Agent Workflow

```
┌─────────────┐
│   Planner   │  (Claude 3.5 Sonnet)
│  Architect  │  • Analyzes task
└──────┬──────┘  • Designs architecture
       │         • Creates step-by-step plan
       ▼
┌─────────────┐
│    Coder    │  (GPT-4o)
│  Developer  │  • Implements plan
└──────┬──────┘  • Writes production code
       │         • Handles edge cases
       ▼
┌─────────────┐
│  Reviewer   │  (o1-mini)
│   Auditor   │  • Reviews code quality
└──────┬──────┘  • Checks security
       │         • Validates correctness
       ▼
   APPROVED? ───No──→ Loop back to Coder (max N times)
       │
      Yes
       ▼
    [END]
```

### Optional Supervisor Mode
When `--use-supervisor` is enabled, an LLM-based supervisor decides routing:
```
START → Supervisor → [Planner|Coder|Reviewer|END]
```

---

## Advanced Features

### 1. Add Tools (Web Search, File Ops, etc.)
```python
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode

tool = DuckDuckGoSearchRun()
llm = get_coder_llm().bind_tools([tool])
```

### 2. Parallel Agent Fan-out
```python
graph.add_edge("planner", "coder")
graph.add_edge("planner", "researcher")
graph.add_edge(["coder", "researcher"], "reviewer")
```

### 3. Human-in-the-Loop
```python
app = graph.compile(interrupt_before=["reviewer"])
```

### 4. Memory with Thread IDs
```python
result = app.invoke(
    state,
    config={"configurable": {"thread_id": "project-1"}}
)
```

---

## Environment Variables

Alternative to CLI args (lower priority):

```bash
export CODER_MODEL=gpt-4-turbo
export REVIEWER_MODEL=o1
export SUPERVISOR_MODEL=gpt-4o-mini
export ENABLE_MEMORY=true
export LANGGRAPH_MEMORY_PATH=agent_memory.db
```

---

## Output Format

The system logs progress and displays:
- **Plan**: Architecture and implementation strategy
- **Code**: Generated production-ready code
- **Review**: Security/quality audit results
- **Final Status**: `approved`, `needs_revision`, or `error`

### Example Output
```
2026-02-04 18:45:00 [INFO] multi_agent: === Multi-Agent System Starting ===
2026-02-04 18:45:00 [INFO] multi_agent: Task: Build a REST API with FastAPI
2026-02-04 18:45:00 [INFO] multi_agent: Planner: claude-3-5-sonnet-20241022
2026-02-04 18:45:05 [INFO] multi_agent: PLANNER: generating plan
2026-02-04 18:45:15 [INFO] multi_agent: [PLANNER] Plan generated (3245 chars)
2026-02-04 18:45:15 [INFO] multi_agent: CODER: generating code
2026-02-04 18:45:35 [INFO] multi_agent: [CODER] Code generated (8932 chars)
2026-02-04 18:45:35 [INFO] multi_agent: REVIEWER: reviewing code
2026-02-04 18:45:50 [INFO] multi_agent: [REVIEWER] APPROVED
2026-02-04 18:45:50 [INFO] multi_agent: === Execution Complete ===
2026-02-04 18:45:50 [INFO] multi_agent: Final status: approved
2026-02-04 18:45:50 [INFO] multi_agent: Code saved to: api.py
```

---

## Troubleshooting

### Missing API Keys
```
Error: ANTHROPIC_API_KEY not set
```
**Solution**: Export API keys before running

### PyYAML Not Found
```
ImportError: PyYAML not installed
```
**Solution**: `pip install pyyaml`

### Max Iterations Reached
```
ROUTER: max iterations (3) reached — ending
```
**Solution**: Increase `--max-iterations` or review task complexity

---

## License

MIT License - Use freely for any project.

---

## Contributing

Contributions welcome! Areas for enhancement:
- Additional agent types (Researcher, Tester, etc.)
- More sophisticated routing logic
- Tool integration examples
- Streaming output support
- Web UI dashboard

---

## Credits

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).
