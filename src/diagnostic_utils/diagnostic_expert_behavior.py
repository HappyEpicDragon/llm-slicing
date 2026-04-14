"""
Expert行为诊断工具
在测试集上分析Expert的资源分配模式
"""

import numpy as np
from src.basic_apis.ppo.ppo_oneshot.global_slicing_env import GlobalSlicingEnv
from src.basic_apis.network_slicing_business.path_manager import PathManager
from omegaconf import OmegaConf


def diagnose_expert(path_manager: PathManager):
    """诊断Expert在测试集上的行为"""

    # 加载配置
    environment_cfg = OmegaConf.load('/root/decision_transformer_slicing/conf/environment/env_oneshot.yaml')

    environment_cfg.env_settings.mode = 'testing'
    environment_cfg.env_settings.model_name = 'diagnostic'
    environment_cfg.env_settings.inside.testing.active_scenario_list = [0]

    # 创建测试环境
    env = GlobalSlicingEnv(env_settings=environment_cfg.env_settings, np_random=None, path_manager=path_manager)

    # 统计变量
    expert_stats = {
        'total_rbgs_allocated': [],
        'hp_rbgs_allocated': [],
        'nhp_rbgs_remaining': [],
        'hp_violations': 0,
        'nhp_violations': 0,
        'timesteps_analyzed': 0
    }

    # 运行1个episode
    obs, info = env.reset()

    print("\n" + "=" * 80)
    print("🔍 Expert Behavior Diagnostic")
    print("=" * 80)

    for step in range(1000):
        # 获取Expert的决策
        expert_action = env._compute_expert_action()

        # 统计Expert分配的RBG数量
        expert_rbgs = np.sum(expert_action > 0)
        expert_stats['total_rbgs_allocated'].append(expert_rbgs)

        # 识别HP用户
        slice_assoc = env.components.slices.ue_assoc
        current_slice_req = env.components.slices.requirements

        hp_users = []
        nhp_users = []
        for s_idx in range(env.num_slices):
            req = current_slice_req.get(f'slice_{s_idx}', {})
            u_indices = np.where(slice_assoc[s_idx] > 0)[0]

            if req.get('priority', 0) > 0:
                hp_users.extend(u_indices.tolist())
            else:
                nhp_users.extend(u_indices.tolist())

        # 统计Expert给HP分配了多少RBG
        hp_rbgs = 0
        for rbg_idx, user_id in enumerate(expert_action):
            if user_id > 0 and (user_id - 1) in hp_users:
                hp_rbgs += 1

        expert_stats['hp_rbgs_allocated'].append(hp_rbgs)
        expert_stats['nhp_rbgs_remaining'].append(env.num_rbgs - hp_rbgs)

        # 执行随机action（NHP部分）
        random_action = np.zeros_like(expert_action)
        for rbg_idx in range(env.num_rbgs):
            if expert_action[rbg_idx] == 0 and len(nhp_users) > 0:
                random_action[rbg_idx] = np.random.choice(nhp_users) + 1

        # 合并Expert和Agent的action
        final_action = expert_action.copy()
        final_action[expert_action == 0] = random_action[expert_action == 0]

        # Step环境
        obs, reward, done, truncated, info = env.step(final_action)

        # 统计违约
        for key in info.keys():
            if key.startswith("drift/slice_"):
                parts = key.split('_')
                if len(parts) >= 2:
                    s_idx = int(parts[1])
                    metric = parts[2] if len(parts) > 2 else None

                    req = current_slice_req.get(f'slice_{s_idx}', {})
                    is_hp = req.get('priority', 0) > 0

                    drift = info[key]
                    if drift < 0:
                        if is_hp:
                            expert_stats['hp_violations'] += 1
                        else:
                            expert_stats['nhp_violations'] += 1

        expert_stats['timesteps_analyzed'] += 1

        # 每200步打印一次
        if step % 200 == 0:
            recent_expert_rbgs = np.mean(expert_stats['total_rbgs_allocated'][-200:]) if len(
                expert_stats['total_rbgs_allocated']) > 0 else 0
            recent_hp_rbgs = np.mean(expert_stats['hp_rbgs_allocated'][-200:]) if len(
                expert_stats['hp_rbgs_allocated']) > 0 else 0

            print(f"\n[Step {step}]")
            print(f"  Expert allocated: {recent_expert_rbgs:.1f} / {env.num_rbgs} RBGs")
            print(f"  HP got: {recent_hp_rbgs:.1f} RBGs ({recent_hp_rbgs / env.num_rbgs * 100:.1f}%)")
            print(f"  NHP remaining: {env.num_rbgs - recent_hp_rbgs:.1f} RBGs")

        if done or truncated:
            break

    # 最终报告
    print("\n" + "=" * 80)
    print("📊 Expert Diagnostic Report")
    print("=" * 80)

    avg_expert_rbgs = np.mean(expert_stats['total_rbgs_allocated'])
    avg_hp_rbgs = np.mean(expert_stats['hp_rbgs_allocated'])
    avg_nhp_rbgs = np.mean(expert_stats['nhp_rbgs_remaining'])

    print(f"\n1. Resource Allocation:")
    print(f"   Total RBGs available: {env.num_rbgs}")
    print(f"   Expert allocated (avg): {avg_expert_rbgs:.2f} RBGs ({avg_expert_rbgs / env.num_rbgs * 100:.1f}%)")
    print(f"   HP received (avg): {avg_hp_rbgs:.2f} RBGs ({avg_hp_rbgs / env.num_rbgs * 100:.1f}%)")
    print(f"   NHP remaining (avg): {avg_nhp_rbgs:.2f} RBGs ({avg_nhp_rbgs / env.num_rbgs * 100:.1f}%)")

    print(f"\n2. Violations:")
    print(f"   HP violations: {expert_stats['hp_violations']}")
    print(f"   NHP violations: {expert_stats['nhp_violations']}")

    hp_viol_rate = expert_stats['hp_violations'] / (expert_stats['timesteps_analyzed'] * 3) * 100
    nhp_viol_rate = expert_stats['nhp_violations'] / (expert_stats['timesteps_analyzed'] * 9) * 100

    print(f"\n3. Violation Rates:")
    print(f"   HP: {hp_viol_rate:.2f}%")
    print(f"   NHP: {nhp_viol_rate:.2f}%")

    # 诊断结论
    print(f"\n4. Diagnosis:")
    if avg_hp_rbgs < env.num_rbgs * 0.3:
        print(f"   ⚠️  Expert只分配了{avg_hp_rbgs / env.num_rbgs * 100:.1f}%的RBG给HP")
        print(f"   → Expert可能过于保守（Capped Demand太低）")

    if hp_viol_rate > 10:
        print(f"   ❌ HP违约率{hp_viol_rate:.1f}%过高")
        print(f"   → Expert策略无法满足HP需求")
        print(f"   → 建议检查Expert的_compute_expert_action逻辑")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    path_manager = PathManager('/root/decision_transformer_slicing/outputs/runs/channel_generality/diagnostics')
    path_manager.root_path = '/root/decision_transformer_slicing'
    diagnose_expert(path_manager)