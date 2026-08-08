# M1 Reproduction Runbook (WSL2 + API mode)

> Repo: `Oasis-AI-Lab/Oasis-AdaMem` · Official code zero-modification: `reference/AdaMEM-official/`
> Execution env: WSL2 Ubuntu-24.04 (`~/adamen-official/`, copied from reference)
> This file is the single authority for "reproduce from scratch" (gate G4-1).

## Why WSL2

- textworld has no Windows wheel (jericho build failure) → WSL2 Linux has ready wheels
- WebShop (M2) requires Python ≤3.10; Linux is less painful
- No GPU on this machine: all model calls go through OpenAI-compatible APIs; no vLLM needed

## Environment Variables (from Phase 1)

| Variable | Value | Description |
|---|---|---|
| `SPLIT` | `eval_in_distribution` / `eval_out_of_distribution` / `train` | ALFWorld split |
| `MEM_TYPE` | empty / `synapse` / `reasoningbank` / `adamem-low` / `adamem-high` | memory mechanism |
| `MODEL_NAME` | `deepseek-chat` | policy model (DeepSeek API) |
| `OPENAI_BASE_IP_ADDR` | `https://api.deepseek.com` | OpenAI-compatible endpoint (**no `/v1`**; code appends it) |
| `OPENAI_API_KEY` | `$DEEPSEEK_API_KEY` | API key (vllm backend also reads this) |
| `BACKEND` | `vllm` (default) | keep default; `openai` branch is for official OpenAI only |
| `EXTRA_MIN_TOKENS` | `0` for DeepSeek; unset/`1` for vLLM | strips vLLM-specific `min_tokens` extra_body |
| `EMB_VLLM_SERVER` | `127.0.0.1` | embedding server host (**hostname only**; utils hardcodes `:8002`) |
| `ALFWORLD_DATA` | `~/.cache/alfworld` | game data root (config uses `$ALFWORLD_DATA/json_2.1.1/...`) |
| `EVAL_BATCH_SIZE` | `140` (seen) / `134` (unseen) | equals game count for deterministic eval |
| `TEST_TIMES` | `3` (default) | paper: 3 runs mean±std |
| `MAX_STEPS` | `50` (default) | max steps per episode |
| `RETRIEVAL_TOPK` | `1` (default) | retrieval k |
| `CONCURRENT_ENV_BATCH_SIZE` | `16` | rate-limit control for API backends |

## Run Commands (inside ~/adamen-official)

```bash
cd ~/adamen-official
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY=$DEEPSEEK_API_KEY
export EXTRA_MIN_TOKENS=0

# Phase 1: no-memory baseline (seen split)
SPLIT=eval_in_distribution EVAL_BATCH_SIZE=140 \
OPENAI_BASE_IP_ADDR=https://api.deepseek.com \
MODEL_NAME=deepseek-chat \
.venv/bin/python -m examples.prompt_agent.gpt4o_alfworld

# Phase 3: memory mechanisms (after index is built)
SPLIT=eval_in_distribution EVAL_BATCH_SIZE=140 MEM_TYPE=synapse ...（same as above）
SPLIT=eval_in_distribution EVAL_BATCH_SIZE=140 MEM_TYPE=reasoningbank ...
SPLIT=eval_in_distribution EVAL_BATCH_SIZE=140 MEM_TYPE=adamem-low ...
SPLIT=eval_in_distribution EVAL_BATCH_SIZE=140 MEM_TYPE=adamem-high ...

# unseen split (generalization)
SPLIT=eval_out_of_distribution EVAL_BATCH_SIZE=134 ...
```

Logs: `logs/alfworld/deepseek-chat/traj_<split>[_<memtype>].json` + `stats_*.txt`

## Data Preparation (done 2026-08-08)

1. `json_2.1.3_tw-pddl.zip` (36.5MB, GitHub release 0.4.2 via gh-proxy.com) → `~/.cache/alfworld/json_2.1.1/` (train 3553 / seen 140 / unseen 134 / valid_train 200)
2. Zip lacks `traj_data.json` → batch-generated minimal files (`{"task_type": "<dir-prefix>"}`, 4027 total) — TW mode only uses task_type for filtering (full traj_data only needed by thor mode)
3. Logic files: `alfworld/data/alfred.pddl` + `alfred.twl2` (raw.githubusercontent.com) → `~/.cache/alfworld/logic/`

## Dependencies (WSL2 venv, Tsinghua PyPI mirror)

```
openai hnswlib numpy 'ray[default]' fastapi uvicorn requests omegaconf pyyaml
tqdm termcolor gymnasium==0.29.1 torch torchvision textworld "textworld[pddl]"
```

Plus system: `apt-get install python3.12-venv build-essential cmake python-is-python3`

NOTE: `pip install -e .` (setup.py) is the verl training path (flash-attn/vllm GPU deps) — **do NOT install for inference**.

## DeepSeek Compat Patch (applied to WSL copy only)

```bash
cd ~/adamen-official
# base_url: protocol-adaptive; extra_body: EXTRA_MIN_TOKENS switch
# (see experiments/m1/patches/deepseek-compat.patch for the canonical diff)
```

## Embedding Service (Phase 2, local CPU)

```bash
# Start inside WSL. CRITICAL: keep wsl.exe alive, otherwise the distro shuts down
# and kills the service. Use Hermes terminal background=true:
#   wsl -d Ubuntu-24.04 -e bash -lc 'cd ~/adamen-official && exec .venv/bin/python experiments/m1/embed_server.py --port 8002 --model Qwen/Qwen3-Embedding-0.6B'
# (nohup + exiting wsl dies)
cd ~/adamen-official
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  .venv/bin/python experiments/m1/embed_server.py --port 8002 --model Qwen/Qwen3-Embedding-0.6B

# Verify
curl -s http://127.0.0.1:8002/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":["hello"]}' | head -c 200
```

- Paper uses Qwen3-Embedding-4B (dim 2560); we use same-family **0.6B** (dim 1024) — near-quality retrieval, CPU-runnable
- **Alias mapping**: official code hardcodes `Qwen/Qwen3-Embedding-4B`; server maps it to 0.6B (`MODEL_ALIASES`), official code untouched
- `EMB_VLLM_SERVER` = hostname only (utils hardcodes `:8002`)
- HNSW index dim follows the embedding model; keep consistent when rebuilding
- Gotcha: new huggingface_hub defaults to xet download protocol (hf-mirror unsupported, 401 crash) — must set `HF_HUB_DISABLE_XET=1`

## Known Pitfalls

- WSL Ubuntu 24.04 needs `apt-get install python3.12-venv` first, else `python3 -m venv` silently fails (pipe tail swallows the error code)
- Tsinghua mirror is slow for big wheels (torch etc., 20-40 min); Aliyun `mirrors.aliyun.com/pypi/simple/` responds faster (0.4s)
- astral.sh is blocked; do NOT use uv install.sh — plain pip inside WSL
- `gpt4o_*.py` script names are historical (support any OpenAI-compatible endpoint via the `BACKEND=vllm` branch)
- Ray worker processes only inherit `PYTHONPATH`, not `sys.path` — always `export PYTHONPATH=/root/adamen-official:/root/adamen-official/agent_system/environments/env_package/alfworld`
- `envs.get_admissible_commands` is a property (list), not a method
- WSL dies when wsl.exe exits (kills nohup children); run long-lived services via Hermes background terminal
