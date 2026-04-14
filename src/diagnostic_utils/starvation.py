"""
诊断：NHP用户为什么完全没有获得服务？

检查以下可能原因：
1. Expert覆盖了所有RBG → 没有剩余资源给NHP
2. RL Agent决策时Action Mask错误 → NHP用户被屏蔽
3. RL学到了"只服务HP"的策略 → 主动忽略NHP
4. Power Scaling导致NHP信号功率为0 → 物理层无法传输
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict


def analyze_starvation_root_cause(episode_data_path: str):
    """分析NHP饿死的根本原因"""

    print("=" * 80)
    print("NHP STARVATION ROOT CAUSE ANALYSIS")
    print("=" * 80)

    # 加载数据
    data = np.load(episode_data_path, allow_pickle=True)

    sched_decision = data["sched_decision"]  # [Steps, Users, RBs]
    pkt_throughputs = data["pkt_throughputs"]
    buffer_occupancies = data["buffer_occupancies"]
    slice_ue_assoc = data["slice_ue_assoc"]
    slice_req = data["slice_req"]

    num_steps = sched_decision.shape[0]
    num_users = sched_decision.shape[1]
    num_rbs = sched_decision.shape[2]

    print(f"\n📋 Dataset Info:")
    print(f"   Steps: {num_steps}")
    print(f"   Users: {num_users}")
    print(f"   Physical RBs: {num_rbs}")

    # ============================================
    # 1. 识别HP和NHP用户
    # ============================================
    hp_users = set()
    nhp_users = set()
    slice_priority_map = {}

    for step in [0]:  # 只看第一步，假设用户分配不变
        for slice_name, req_dict in slice_req[step].items():
            if not req_dict:
                continue

            s_idx = int(slice_name.split('_')[1])
            prio = req_dict.get('priority', 0)
            slice_priority_map[s_idx] = prio

            ue_indices = np.where(slice_ue_assoc[step][s_idx] > 0)[0]

            if prio > 0:
                hp_users.update(ue_indices.tolist())
            else:
                nhp_users.update(ue_indices.tolist())

    hp_users = sorted(list(hp_users))
    nhp_users = sorted(list(nhp_users))

    print(f"\n👥 User Classification:")
    print(f"   HP Users ({len(hp_users)}): {hp_users}")
    print(f"   NHP Users ({len(nhp_users)}): {nhp_users}")

    # ============================================
    # 2. 统计资源分配
    # ============================================
    print(f"\n📊 Resource Allocation Analysis:")

    # 每个用户获得的RB数量（时间平均）
    rb_allocated_per_user = np.sum(sched_decision > 0, axis=(0, 2))  # [Users]

    total_hp_rbs = np.sum(rb_allocated_per_user[hp_users])
    total_nhp_rbs = np.sum(rb_allocated_per_user[nhp_users])
    total_rbs = num_steps * num_rbs

    print(f"   Total RB-steps available: {total_rbs}")
    print(f"   HP allocated: {total_hp_rbs} ({total_hp_rbs / total_rbs * 100:.1f}%)")
    print(f"   NHP allocated: {total_nhp_rbs} ({total_nhp_rbs / total_rbs * 100:.1f}%)")
    print(f"   Unused: {total_rbs - total_hp_rbs - total_nhp_rbs}")

    if total_nhp_rbs == 0:
        print(f"\n   🚨 CRITICAL: NHP received ZERO RBs in entire episode!")

    # 每步的RB使用情况
    rbs_used_per_step = np.sum(sched_decision > 0, axis=(1, 2))  # [Steps]
    avg_rbs_used = np.mean(rbs_used_per_step)

    print(f"\n   Average RBs used per step: {avg_rbs_used:.1f} / {num_rbs}")
    print(f"   Utilization: {avg_rbs_used / num_rbs * 100:.1f}%")

    # ============================================
    # 3. 分析NHP用户的需求
    # ============================================
    print(f"\n📋 NHP User Demand Analysis:")

    for u in nhp_users:
        # 统计该用户的Buffer情况
        avg_buffer = np.mean(buffer_occupancies[:, u])
        max_buffer = np.max(buffer_occupancies[:, u])

        # 统计该用户获得的吞吐量
        total_thr = np.sum(pkt_throughputs[:, u])

        # 统计该用户获得的RB数
        total_rbs = np.sum(sched_decision[:, u, :] > 0)

        print(f"\n   User {u}:")
        print(f"      Buffer: avg={avg_buffer:.2%}, max={max_buffer:.2%}")
        print(f"      Total throughput: {total_thr:.2f} packets")
        print(f"      Total RBs: {total_rbs}")

        if avg_buffer > 0.5 and total_rbs == 0:
            print(f"      🚨 High buffer but ZERO allocation!")

    # ============================================
    # 4. 检查是否是Power Scaling问题
    # ============================================
    print(f"\n⚡ Power Analysis:")

    # 检查是否有功率信息
    if 'target_cell_power' in data:
        target_cell_power = data['target_cell_power']  # [Steps, Users, RBs] 或其他维度

        # 检查NHP用户的信道增益
        for u in nhp_users[:3]:  # 只看前3个NHP用户
            if target_cell_power.ndim == 3:
                user_power = target_cell_power[:, u, :]
            elif target_cell_power.ndim == 4:
                user_power = target_cell_power[:, u, 0, :]  # 假设第三维是1
            else:
                continue

            avg_power = np.mean(user_power)
            min_power = np.min(user_power)

            print(f"   User {u}: avg_power={avg_power:.2e}, min_power={min_power:.2e}")

            if avg_power < 1e-20:
                print(f"      ⚠️  Extremely low power - channel dead zone?")

    # ============================================
    # 5. 逐步分析：从RL决策到物理层执行
    # ============================================
    print(f"\n🔍 Step-by-Step Analysis (Sample: Step 500):")

    sample_step = 500

    # 5.1 RL决策
    rl_decision = sched_decision[sample_step]  # [Users, RBs]

    hp_rbs_in_decision = np.sum(rl_decision[hp_users, :] > 0)
    nhp_rbs_in_decision = np.sum(rl_decision[nhp_users, :] > 0)

    print(f"\n   1. RL Decision Output:")
    print(f"      HP RBs: {hp_rbs_in_decision}")
    print(f"      NHP RBs: {nhp_rbs_in_decision}")

    # 5.2 检查NHP用户是否被分配但功率为0
    nhp_has_allocation = []
    for u in nhp_users:
        if np.sum(rl_decision[u, :] > 0) > 0:
            nhp_has_allocation.append(u)

    if len(nhp_has_allocation) > 0:
        print(f"\n   2. NHP users with allocation: {nhp_has_allocation}")
        print(f"      → But got zero throughput → Power scaling issue?")
    else:
        print(f"\n   2. NHP users with allocation: NONE")
        print(f"      → RL agent is NOT allocating RBs to NHP at all!")

    # ============================================
    # 6. 检查Expert Override（如果是testing模式）
    # ============================================
    print(f"\n🤖 Expert Override Analysis:")

    # 如果存在agent_action字段，说明有expert override
    if 'agent_action' in data:
        agent_actions = data['agent_action']  # [Steps, RBGs]

        # 检查Expert是否覆盖了所有RBG
        sample_action = agent_actions[sample_step]

        # 统计每个user被分配了多少RBG
        rbg_size = num_rbs // 27  # 假设27个RBG
        user_rbg_count = defaultdict(int)

        for rbg_idx, user_id in enumerate(sample_action):
            if user_id > 0:
                user_rbg_count[user_id - 1] += 1

        hp_rbgs = sum(user_rbg_count[u] for u in hp_users if u in user_rbg_count)
        nhp_rbgs = sum(user_rbg_count[u] for u in nhp_users if u in user_rbg_count)
        unallocated_rbgs = 27 - hp_rbgs - nhp_rbgs

        print(f"   Agent Action (RBG level):")
        print(f"      HP RBGs: {hp_rbgs} / 27")
        print(f"      NHP RBGs: {nhp_rbgs} / 27")
        print(f"      Unallocated: {unallocated_rbgs} / 27")

        if nhp_rbgs == 0 and hp_rbgs >= 25:
            print(f"\n      🚨 Expert/Agent allocates all RBGs to HP!")
            print(f"         → This is the root cause of NHP starvation")

    # ============================================
    # 7. 总结诊断结论
    # ============================================
    print(f"\n" + "=" * 80)
    print("DIAGNOSTIC CONCLUSION")
    print("=" * 80)

    if total_nhp_rbs == 0:
        print(f"\n✅ ROOT CAUSE IDENTIFIED: NHP received ZERO RBs")

        if 'agent_action' in data and nhp_rbgs == 0:
            print(f"\n   Primary Cause: Agent/Expert Decision")
            print(f"      → RL agent learned to allocate all RBs to HP")
            print(f"      → OR Expert override covers all RBGs with HP users")
            print(f"\n   Solution:")
            print(f"      1. Check if Expert is too aggressive in HP protection")
            print(f"      2. Add explicit NHP service reward")
            print(f"      3. Use soft constraints instead of hard override")
        else:
            print(f"\n   Primary Cause: Power Scaling or Physical Layer")
            print(f"      → RBs are allocated but power = 0")
            print(f"      → Check power scaling logic")

    print("=" * 80)

    # 保存详细数据
    output_dir = Path("diagnostic")
    output_dir.mkdir(exist_ok=True)

    report = {
        'summary': {
            'hp_users': hp_users,
            'nhp_users': nhp_users,
            'total_hp_rbs': int(total_hp_rbs),
            'total_nhp_rbs': int(total_nhp_rbs),
            'avg_rbs_used_per_step': float(avg_rbs_used),
            'utilization': float(avg_rbs_used / num_rbs)
        }
    }

    with open(output_dir / "starvation_root_cause.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Report saved to: diagnostic/starvation_root_cause.json")


if __name__ == "__main__":
    analyze_starvation_root_cause('/root/decision_transformer_slicing/data/channel_generality/ppo_oneshot/metric_raw/scenario_0/ep_61.npz')