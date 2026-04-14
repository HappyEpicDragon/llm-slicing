"""
诊断脚本：检测"丢包率陷阱"现象

检测逻辑：
1. 找到NHP用户从"饿死"转为"获得资源"的时刻
2. 追踪恢复过程中三类指标的演化
3. 验证假设：吞吐量和时延可以快速恢复，但丢包率仍然违约
"""

import numpy as np
import json
from collections import defaultdict
from pathlib import Path


def analyze_packet_loss_trap(episode_data_path: str):
    """
    分析丢包率陷阱现象

    Args:
        episode_data_path: .npz文件路径
    """
    print("=" * 80)
    print("PACKET LOSS TRAP ANALYSIS")
    print("=" * 80)

    # 加载数据
    data = np.load(episode_data_path, allow_pickle=True)

    pkt_throughputs = data["pkt_throughputs"]  # [Steps, Users]
    buffer_latencies = data["buffer_latencies"]  # [Steps, Users]
    dropped_pkts = data["dropped_pkts"]  # [Steps, Users]
    pkt_incoming = data["pkt_incoming"]  # [Steps, Users]
    slice_ue_assoc = data["slice_ue_assoc"]  # [Steps, Slices, Users]
    slice_req = data["slice_req"]  # [Steps]

    num_steps = pkt_throughputs.shape[0]
    num_users = pkt_throughputs.shape[1]

    # ============================================
    # 1. 识别NHP用户
    # ============================================
    nhp_users = set()
    for step in range(num_steps):
        for s_idx, req_dict in enumerate(slice_req[step].values()):
            if req_dict and req_dict.get('priority', 0) == 0:
                ue_indices = np.where(slice_ue_assoc[step][s_idx] > 0)[0]
                nhp_users.update(ue_indices.tolist())

    print(f"\n📋 Found {len(nhp_users)} NHP users")

    # ============================================
    # 2. 检测"饿死→恢复"事件
    # ============================================
    starvation_recovery_events = []

    STARVATION_THRESHOLD = 1e-9  # 吞吐量几乎为0
    RECOVERY_THRESHOLD = 1e-6  # 吞吐量恢复到有意义的值
    MIN_STARVATION_DURATION = 50  # 至少饿死50ms才算

    for u in nhp_users:
        starved_start = None

        for step in range(num_steps):
            thr = pkt_throughputs[step, u]

            # 检测饿死开始
            if thr < STARVATION_THRESHOLD and starved_start is None:
                starved_start = step

            # 检测恢复
            elif thr > RECOVERY_THRESHOLD and starved_start is not None:
                duration = step - starved_start
                if duration >= MIN_STARVATION_DURATION:
                    starvation_recovery_events.append({
                        'user': u,
                        'starved_start': starved_start,
                        'recovery_start': step,
                        'starvation_duration': duration
                    })
                starved_start = None

    print(f"\n🔍 Detected {len(starvation_recovery_events)} starvation→recovery events")

    if len(starvation_recovery_events) == 0:
        print("⚠️  No recovery events found. Cannot analyze trap effect.")
        return

    # ============================================
    # 3. 分析每个恢复事件的指标演化
    # ============================================
    TRACKING_WINDOW = 200  # 恢复后追踪200ms

    trap_cases = []

    for event in starvation_recovery_events[:10]:  # 只分析前10个事件
        u = event['user']
        t0 = event['recovery_start']

        if t0 + TRACKING_WINDOW >= num_steps:
            continue

        # 计算恢复过程中的指标
        thr_before = pkt_throughputs[t0 - 1, u] if t0 > 0 else 0
        lat_before = buffer_latencies[t0 - 1, u] if t0 > 0 else 0

        # 追踪恢复曲线
        thr_curve = pkt_throughputs[t0:t0 + TRACKING_WINDOW, u]
        lat_curve = buffer_latencies[t0:t0 + TRACKING_WINDOW, u]

        # 计算累积丢包率
        # 丢包率 = 累积丢包 / (累积丢包 + 累积发送)
        cumsum_dropped = np.cumsum(dropped_pkts[t0:t0 + TRACKING_WINDOW, u])
        cumsum_sent = np.cumsum(pkt_throughputs[t0:t0 + TRACKING_WINDOW, u])

        packet_loss_rate = np.zeros_like(cumsum_dropped)
        for i in range(len(cumsum_dropped)):
            total = cumsum_dropped[i] + cumsum_sent[i]
            if total > 0:
                packet_loss_rate[i] = cumsum_dropped[i] / total

        # 检测恢复时间
        def find_recovery_time(curve, threshold):
            """找到指标恢复到阈值以下的时间"""
            for i, val in enumerate(curve):
                if val < threshold:
                    return i
            return len(curve)

        # 吞吐量恢复时间（假设SLA是10 Mbps = 10e6 bits/s = 10e3 bits/ms）
        thr_sla = 1.0  # 简化：假设恢复到>1 packet/ms就算满足
        thr_recovery_time = find_recovery_time(np.abs(thr_sla - thr_curve), thr_sla)

        # 时延恢复时间（假设SLA是<10ms）
        lat_sla = 10.0
        lat_recovery_time = find_recovery_time(lat_curve, lat_sla)

        # 丢包率恢复时间（假设SLA是<0.1%）
        plr_sla = 0.001
        plr_recovery_time = find_recovery_time(packet_loss_rate, plr_sla)

        # 判断是否存在"陷阱"
        is_trap = (plr_recovery_time > 2 * max(thr_recovery_time, lat_recovery_time))

        trap_cases.append({
            'user': u,
            'event': event,
            'thr_recovery_time': thr_recovery_time,
            'lat_recovery_time': lat_recovery_time,
            'plr_recovery_time': plr_recovery_time,
            'is_trap': is_trap,
            'thr_curve': thr_curve.tolist(),
            'lat_curve': lat_curve.tolist(),
            'plr_curve': packet_loss_rate.tolist()
        })

    # ============================================
    # 4. 统计结果
    # ============================================
    num_traps = sum(1 for case in trap_cases if case['is_trap'])

    print(f"\n📊 Recovery Analysis Results:")
    print(f"   Total analyzed events: {len(trap_cases)}")
    print(f"   Trap cases detected: {num_traps} ({num_traps / len(trap_cases) * 100:.1f}%)")

    print(f"\n⏱️  Average Recovery Times:")
    avg_thr_recovery = np.mean([c['thr_recovery_time'] for c in trap_cases])
    avg_lat_recovery = np.mean([c['lat_recovery_time'] for c in trap_cases])
    avg_plr_recovery = np.mean([c['plr_recovery_time'] for c in trap_cases])

    print(f"   Throughput: {avg_thr_recovery:.1f} ms")
    print(f"   Latency:    {avg_lat_recovery:.1f} ms")
    print(f"   Packet Loss Rate: {avg_plr_recovery:.1f} ms")

    if avg_plr_recovery > 2 * max(avg_thr_recovery, avg_lat_recovery):
        print(f"\n🚨 TRAP CONFIRMED!")
        print(f"   PLR recovery is {avg_plr_recovery / max(avg_thr_recovery, avg_lat_recovery):.1f}x slower")
        print(f"   → This creates a negative reward trap for RL agent")

    # ============================================
    # 5. 展示典型案例
    # ============================================
    if len(trap_cases) > 0:
        print(f"\n📋 Example Trap Cases:")
        for i, case in enumerate(trap_cases[:3]):
            if case['is_trap']:
                print(f"\n   Case {i + 1}: User {case['user']}")
                print(f"      Starvation duration: {case['event']['starvation_duration']} ms")
                print(f"      Recovery times:")
                print(f"         Throughput: {case['thr_recovery_time']} ms ✅")
                print(f"         Latency:    {case['lat_recovery_time']} ms ✅")
                print(f"         PLR:        {case['plr_recovery_time']} ms ❌ (TRAP!)")

    # ============================================
    # 6. 保存详细数据
    # ============================================
    output_dir = Path("diagnostic")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "packet_loss_trap_analysis.json", 'w') as f:
        json.dump({
            'summary': {
                'total_events': len(trap_cases),
                'trap_cases': num_traps,
                'trap_rate': num_traps / len(trap_cases) if trap_cases else 0,
                'avg_recovery_times': {
                    'throughput_ms': float(avg_thr_recovery),
                    'latency_ms': float(avg_lat_recovery),
                    'packet_loss_rate_ms': float(avg_plr_recovery)
                }
            },
            'cases': trap_cases
        }, f, indent=2)

    print(f"\n💾 Detailed analysis saved to: diagnostic/packet_loss_trap_analysis.json")
    print("=" * 80)


def analyze_persistent_violations(episode_data_path: str):
    """
    分析哪些切片的丢包率长期违约
    """
    print("\n" + "=" * 80)
    print("PERSISTENT VIOLATION ANALYSIS")
    print("=" * 80)

    data = np.load(episode_data_path, allow_pickle=True)

    dropped_pkts = data["dropped_pkts"]
    pkt_throughputs = data["pkt_throughputs"]
    slice_ue_assoc = data["slice_ue_assoc"]
    slice_req = data["slice_req"]

    num_steps = dropped_pkts.shape[0]

    # 按切片统计丢包率违约情况
    slice_violation_stats = defaultdict(lambda: {
        'total_steps': 0,
        'violation_steps': 0,
        'users': set()
    })

    for step in range(num_steps):
        for s_idx, (slice_name, req_dict) in enumerate(slice_req[step].items()):
            if not req_dict:
                continue

            # 获取该切片的reliability SLA
            rel_sla = None
            if 'parameters' in req_dict:
                for param in req_dict['parameters'].values():
                    if param.get('name') == 'reliability':
                        rel_sla = param.get('value', 99.9)  # 默认99.9%
                        break

            if rel_sla is None:
                continue

            # 计算该切片的丢包率
            ue_indices = np.where(slice_ue_assoc[step][s_idx] > 0)[0]
            slice_violation_stats[slice_name]['users'].update(ue_indices.tolist())
            slice_violation_stats[slice_name]['total_steps'] += 1

            if len(ue_indices) > 0:
                slice_dropped = np.sum(dropped_pkts[:step + 1, ue_indices])
                slice_sent = np.sum(pkt_throughputs[:step + 1, ue_indices])

                if slice_dropped + slice_sent > 0:
                    plr = slice_dropped / (slice_dropped + slice_sent)
                    # 丢包率 > (100 - reliability)/100 就是违约
                    threshold = (100 - rel_sla) / 100

                    if plr > threshold:
                        slice_violation_stats[slice_name]['violation_steps'] += 1

    print(f"\n📊 Per-Slice Violation Statistics:")
    for slice_name, stats in sorted(slice_violation_stats.items()):
        if stats['total_steps'] > 0:
            violation_rate = stats['violation_steps'] / stats['total_steps']
            print(f"\n   {slice_name}:")
            print(f"      Violation rate: {violation_rate * 100:.1f}%")
            print(f"      Users: {len(stats['users'])}")

            if violation_rate > 0.8:
                print(f"      ⚠️  PERSISTENT VIOLATION (>80% of time)")

    print("=" * 80)


if __name__ == "__main__":

    episode_path = '/root/decision_transformer_slicing/data/channel_generality/ppo_oneshot/metric_raw/scenario_0/ep_61.npz'

    # 运行两个分析
    analyze_packet_loss_trap(episode_path)
    analyze_persistent_violations(episode_path)