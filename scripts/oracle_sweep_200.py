"""
Oracle sweep on ALL 200 scenarios (11 codebooks × RR × 1 episode each).
Scenarios 0-9 have rich channel data; 10-199 have 1 episode only.

Outputs: per-scenario best codebook + observation fingerprints for generalization analysis.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/oracle_sweep_200.py
"""
import os, sys, json, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_SLICES = 5
INTRA = 0  # RR


def _compute_step_metrics(info, num_slices=NUM_SLICES):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        if min(drifts) < 0:
            if is_hp:
                hp_viols += 1
            else:
                nhp_viols += 1
            neg = [d for d in drifts if d < 0]
            if is_hp:
                hp_dist += min(neg)
            else:
                nhp_dist += min(neg)
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_scenario_batch(scenario_ids, n_eps=1):
    """Run Oracle sweep for a batch of scenarios."""
    import torch
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())

    results = {}

    for scen_id in scenario_ids:
        scen_results = {}
        for dim0 in range(11):
            action = np.array([[dim0] + [INTRA] * NUM_SLICES], dtype=np.int32)

            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.model_name = f"scenario_{scen_id}"
            cfg.inside.testing.active_scenario_list = [scen_id]
            cfg.inside.testing.init_scenario_episode = 0
            cfg.inside.testing.max_scenario_episodes = n_eps

            def _make(c=cfg):
                def _init():
                    return HierarchicalSlicingEnv(c, np.random.default_rng(42), pm)
                return _init

            try:
                env = DummyVecEnv([_make()])
                env.reset()

                hp_viol_sum = nhp_viol_sum = 0
                hp_dist_sum = nhp_dist_sum = 0.0
                hp_active_t = nhp_active_t = 0
                obs_fingerprint = []
                done = False

                step_count = 0
                while not done:
                    next_obs, rewards, dones, infos = env.step(action)
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    hd, nd, hv, nv, ha, na = _compute_step_metrics(info)
                    hp_dist_sum += hd; nhp_dist_sum += nd
                    hp_viol_sum += hv; nhp_viol_sum += nv
                    hp_active_t += ha; nhp_active_t += na

                    if step_count < 20 and dim0 == 0:
                        obs_fingerprint.append(next_obs['inter_feat'][0].copy())

                    step_count += 1
                    if dones[0]:
                        done = True

                env.close()

                hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
                nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
                hp_d = hp_dist_sum / hp_active_t if hp_active_t > 0 else 0.0
                nhp_d = nhp_dist_sum / nhp_active_t if nhp_active_t > 0 else 0.0

                scen_results[dim0] = {
                    "hp_viol": hp_v, "nhp_viol": nhp_v,
                    "hp_dist": hp_d, "nhp_dist": nhp_d,
                    "combined": hp_v + nhp_v,
                }

                if dim0 == 0 and obs_fingerprint:
                    scen_results["obs_fingerprint"] = np.array(obs_fingerprint).mean(axis=0).tolist()

            except Exception as e:
                scen_results[dim0] = {"error": str(e)}

        results[scen_id] = scen_results

    return results


def main():
    all_scenarios = list(range(200))
    n_workers = 10

    chunk_size = (len(all_scenarios) + n_workers - 1) // n_workers
    chunks = [all_scenarios[i:i+chunk_size] for i in range(0, len(all_scenarios), chunk_size)]

    total_runs = len(all_scenarios) * 11
    print(f"Oracle sweep on 200 scenarios: {total_runs} runs (11 codebooks × 200 scenarios × 1 ep)")
    print(f"Workers: {n_workers}, chunks: {[len(c) for c in chunks]}")

    t0 = time.time()
    all_results = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for chunk in chunks:
            fut = pool.submit(run_scenario_batch, chunk, n_eps=1)
            futures[fut] = chunk

        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                res = fut.result()
                all_results.update(res)
                elapsed = time.time() - t0
                print(f"  [{elapsed/60:.1f}min] Done: {len(res)} scenarios (S{min(chunk)}..S{max(chunk)})")
            except Exception as e:
                print(f"  FAILED chunk S{min(chunk)}..S{max(chunk)}: {e}")
                import traceback; traceback.print_exc()

    total_time = time.time() - t0
    print(f"\nDone in {total_time/60:.1f} min")

    save_dir = "data/channel_generality/oracle_200"
    os.makedirs(save_dir, exist_ok=True)

    summary = {}
    for scen_id in sorted(all_results.keys()):
        sr = all_results[scen_id]
        best_d0, best_c = None, float("inf")
        codebook_scores = {}
        for d0 in range(11):
            if d0 in sr and "combined" in sr[d0]:
                c = sr[d0]["combined"]
                codebook_scores[d0] = c
                if c < best_c:
                    best_c = c
                    best_d0 = d0

        obs_fp = sr.get("obs_fingerprint", None)
        summary[scen_id] = {
            "best_codebook": best_d0,
            "best_combined": best_c,
            "all_codebooks": codebook_scores,
            "obs_fingerprint": obs_fp,
        }

    with open(os.path.join(save_dir, "summary_200.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print overview
    best_cbs = [summary[s]["best_codebook"] for s in sorted(summary.keys()) if summary[s]["best_codebook"] is not None]
    print(f"\nBest codebook distribution across 200 scenarios:")
    from collections import Counter
    for cb, cnt in sorted(Counter(best_cbs).items()):
        print(f"  act{cb}: {cnt} scenarios ({cnt/len(best_cbs)*100:.1f}%)")

    print(f"\nResults saved to {save_dir}/summary_200.json")


if __name__ == "__main__":
    main()
