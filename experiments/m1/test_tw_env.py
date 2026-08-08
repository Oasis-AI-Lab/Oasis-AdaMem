# Phase 0 gate G0-1 verification: ALFWorld TW env reset/step smoke test
# Usage: ALFWORLD_DATA=~/.cache/alfworld .venv/bin/python test_tw_env.py [env_num]
import os, sys, time

env_num = int(sys.argv[1]) if len(sys.argv) > 1 else 2
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

# Repo root = two levels above this script (experiments/m1 -> repo root)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NOTE: ray worker processes only inherit PYTHONPATH, not sys.path!
# At runtime you must export PYTHONPATH=REPO_ROOT:REPO_ROOT/agent_system/environments/env_package/alfworld
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "agent_system/environments/env_package/alfworld"))

from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
from agent_system.environments.env_manager import AlfWorldEnvironmentManager

alf_config_path = os.path.join(
    REPO_ROOT,
    "agent_system/environments/env_package/alfworld/configs/config_tw.yaml",
)

print(f"[1/4] Building envs (n={env_num}, split=eval_in_distribution) ...")
t0 = time.time()
envs = build_alfworld_envs(
    alf_config_path, seed=1, env_num=env_num, group_n=1, is_train=False,
    env_kwargs={"eval_dataset": "eval_in_distribution"},
    resources_per_worker={"num_cpus": 0.05, "num_gpus": 0.0},
    start_idx=0,
)
print(f"      built in {time.time()-t0:.1f}s")

mgr = AlfWorldEnvironmentManager(envs, alfworld_projection, "alfworld/AlfredThorEnv")
print("[2/4] reset ...")
obs, infos = mgr.reset({})
print(f"      obs texts: {len(obs['text'])}")
for i, t in enumerate(obs["text"]):
    print(f"      --- env {i} (first 300 chars) ---")
    print("      " + t[:300].replace("\n", "\n      "))
    break

print("[3/4] step with a valid action ...")
# get_admissible_commands is a property (List[List[str]]), not a method
adm = envs.get_admissible_commands
print(f"      admissible[0][:5] = {adm[0][:5]}")
actions = [a[0] if a else "look" for a in adm]
next_obs, rewards, dones, infos = mgr.step(actions)
print(f"      rewards={rewards.tolist() if hasattr(rewards,'tolist') else rewards} dones={dones}")

print("[4/4] one more step (inventory) ...")
actions = ["inventory"] * env_num
next_obs, rewards, dones, infos = mgr.step(actions)
print(f"      rewards={rewards} dones={dones}")

print("\nG0-1 PASS: environment reset/step OK")
