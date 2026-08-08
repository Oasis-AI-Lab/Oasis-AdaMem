# AdaMEM M1 Reproduction Plan: Training-Free Inference Pipeline (ALFWorld Focus)

> **Project**: Oasis AdaMem (repo `Oasis-AI-Lab/Oasis-AdaMem`)
> **Paper**: AdaMEM: Test-Time Adaptive Memory for Language Agents (ICML 2026, arXiv:2606.05684)
> **Official code**: `reference/AdaMEM-official/` (yunx-z/AdaMEM, cloned, zero modification)
> **Date**: 2026-08-08

## Goal

On a **GPU-less Windows machine**, using an OpenAI-compatible API for the policy model and a local CPU embedding service, run the official AdaMEM code across 5 memory mechanisms (`no-memory` / `synapse` / `reasoningbank` / `adamem-low` / `adamem-high`) on ALFWorld **seen + unseen** splits, reproduce the relative trends of the paper's Table 1, and deliver a credible, reusable experimental harness.

**Strategic position**: M1 is the foundation. The official code is public, so we do not reimplement from scratch — we build a harness that can run, modify, and quantitatively compare. M2 completes WebShop/HotpotQA and STEP-MFT data collection; M3 enters innovation (direction list at the end).

## Hard Constraints & Feasibility (verified)

| Constraint | Status | Mitigation |
|---|---|---|
| No GPU (no nvidia-smi) | CPU-only machine | All model calls in official code are **OpenAI-compatible API** (policy → `OPENAI_BASE_IP_ADDR`, embedding → `EMB_VLLM_SERVER:8002/v1/embeddings`); point to a remote API |
| LLM service | needs OpenAI-compatible endpoint | DeepSeek API (the only key available); official `examples/` are API-mode by design |
| Embedding service | paper uses Qwen3-Embedding-4B (not feasible on CPU) | Local CPU OpenAI-compatible embedding server: **Qwen3-Embedding-0.6B** (same family, dim 1024), FastAPI wrapper; server-side alias maps 4B→0.6B so official code stays untouched |
| Training trajectories | official code does NOT publish them | Collect via no-memory ReAct rollouts on ALFWorld `train` split (paper: 1,596 successful trajectories) |
| Python env | no conda; uv available | `uv venv`; WebShop needs py3.10 (M2) |
| GPU deps (flash-attn/vllm/verl) | not installable locally | **Not installed.** Minimal inference deps only: openai, hnswlib, numpy, ray, fastapi, uvicorn, transformers |

## Milestone Structure (DoD / Gates / Timebox)

### Phase 0: Environment Setup + Minimal Dependencies (⏱ 0.5 day)

**Tasks**
1. WSL2 Ubuntu-24.04 venv (system python3.12 after `apt-get install python3.12-venv`), minimal inference deps via Tsinghua PyPI mirror
2. Install ALFWorld data: `json_2.1.3_tw-pddl.zip` (GitHub release 0.4.2 via gh-proxy) → `~/.cache/alfworld/json_2.1.1/`; generate minimal `traj_data.json` (`{"task_type": "<dir-prefix>"}`); download logic files (`alfred.pddl`, `alfred.twl2`)
3. Install `textworld[pddl]` (fast-downward, needs build-essential + cmake + `python-is-python3`)
4. Verify env reset/step via `experiments/m1/test_tw_env.py`

**DoD**: `test_tw_env.py` prints `G0-1 PASS`; all imports OK.

**Gates (no green, no forward)**
- G0-1: ALFWorld TW env reset/step/admissible actions work
- G0-2: `import hnswlib, ray, openai, textworld` succeed in venv

**Fallbacks (if G0 fails)**: WSL2 → Docker → rented Linux box (AutoDL, hourly)

### Phase 1: LLM Pipeline + no-memory Baseline (⏱ 1 day)

**Tasks**
1. Configure `OPENAI_BASE_IP_ADDR=https://api.deepseek.com` (no `/v1`; code appends it), `MODEL_NAME=deepseek-chat`, `EXTRA_MIN_TOKENS=0` (DeepSeek mode), `CONCURRENT_ENV_BATCH_SIZE=16` (rate limits)
2. Run `no-memory` ReAct on seen split (140 tasks, `EVAL_BATCH_SIZE=140`)
3. Spot-check logs: traj JSON structure complete (steps/curr_prompt/curr_action/won)

**DoD**: seen-split baseline number + token stats, reproducible logs.

**Gates**
- G1-1: success rate ≥ 35% (paper Qwen3-4B: 45.2±1.8; API model may deviate, below 35% means a broken pipeline)
- G1-2: every trajectory has full prompt history and final win/loss

### Phase 2: Embedding Service + Long-Term Memory Index (⏱ 1 day)

**Tasks**
1. Local FastAPI `/v1/embeddings` (Qwen3-Embedding-0.6B, CPU, batch 256), port 8002; alias 4B→0.6B
2. Run `train` split (150 envs, multiple rounds) to collect successful trajectories → `logs/alfworld_old/<model>/traj_train.json`
3. `build_index.py --dataset_name alfworld --base_model_name <model> --correct_only` → HNSW index
4. Verify `get_top_k_memories` returns semantically relevant experiences

**DoD**: HNSW index built (successful entries ≥ 500); retrieval service usable.

**Gates**
- G2-1: 3/5 manual spot-checks of top-1 retrieval are semantically relevant
- G2-2: embedding service throughput ≥ 5 req/s (CPU batched)

### Phase 3: Full Comparison of Four Memory Mechanisms (⏱ 1.5–2 days)

**Tasks**
1. Run `synapse`, `reasoningbank`, `adamem-low`, `adamem-high` (seen split, `RETRIEVAL_TOPK=1`)
2. Run `unseen` split (134 tasks): `no-memory` + `adamem-low` (generalization trend)
3. Aggregate: success rate, tokens/task, strategy updates/task (vs paper Tables 1/5/6)

**DoD**: 5-mechanism × seen comparison table + unseen generalization table + archived logs.

**Gates**
- G3-1: **trend correct** — `adamem-low ≥ synapse ≥ no-memory` on ALFWorld (relative trend; absolute numbers may deviate)
- G3-2: `adamem-low` beats `no-memory` on unseen (paper: +11.4 pts direction)
- G3-3: ≥ 1 success case with a full "strategy refresh corrects a flawed plan" trajectory (paper Appendix C.2 style)

### Phase 4: Report + Reproducible Runbook (⏱ 0.5 day)

**Tasks**
1. Write `experiments/m1/REPORT.md`: number tables, gap analysis vs paper (model/embedding attribution), log index
2. Finalize `experiments/m1/RUNBOOK.md` (env config templates, dep list, one-command docs)
3. Commit all artifacts to `Oasis-AI-Lab/Oasis-AdaMem`

**DoD**: report + one-command reproduction docs; clean commit.

**Gates**
- G4-1: fresh venv can reproduce Phase 1 baseline following the docs
- G4-2: every number in REPORT.md traces to a log file

## M2 Outlook (after M1 all green)

- WebShop (uv py3.10 env; negative-transfer case — the key battleground for dynamic vs static retrieval)
- HotpotQA (cross-episode agentic search)
- STEP-MFT data collection (`adamem-high` on train split → `filter_sft_data.py` dual filter); training needs GPU → rent cloud

## M3 Innovation Candidates (after reproduction is credible; stackable)

1. **Strategy Inertia attack**: refresh decision from pure model self-eval to external-signal hybrid (repeat-action detection / admissible-action mutation / no-progress detection forcing refresh); negative-example training of refresh decisions
2. **Pitfall Memory**: encode failed trajectories as taboos/pitfalls (paper Appendix E.2 self-acknowledged open problem)
3. **Semantic action equivalence**: replace exact string match `a_t ≠ a'_t` with embedding distance
4. **Real-scenario transfer**: generalize the memory layer into WAM framework / real workflows

## Risk Register

| # | Risk | Prob | Impact | Mitigation |
|---|---|---|---|---|
| R1 | ALFWorld/textworld unusable on Windows | Med | High | WSL2 → Docker → AutoDL cloud |
| R2 | 0.6B embedding quality below paper's 4B | Med | Med | Note in report; rent GPU for 4B index if needed |
| R3 | API model behavior differs from Qwen3-4B | High | Low | Validate by relative mechanism trends, not absolute numbers |
| R4 | Official code dependency conflicts | Med | Med | Minimal inference dep set; skip verl training path |
| R5 | Train-split trajectory collection slow (150 envs × rounds) | Med | Med | Parallel sampling; decouple index build from collection |

## DeepSeek Compatibility (implemented 2026-08-08)

- Patch `experiments/m1/patches/deepseek-compat.patch` (8 hunks: 4× base_url protocol-adaptive, 4× EXTRA_MIN_TOKENS switch), applied to WSL run copy; official repo stays zero-diff
- Env for DeepSeek mode: `OPENAI_BASE_IP_ADDR=https://api.deepseek.com`, `MODEL_NAME=deepseek-chat`, `EXTRA_MIN_TOKENS=0`, `CONCURRENT_ENV_BATCH_SIZE=16`
- Estimated cost for full M1: $40–60 at deepseek-chat list prices

## Pending User Input

1. **API key**: `DEEPSEEK_API_KEY` (Phase 1 hard dependency)
