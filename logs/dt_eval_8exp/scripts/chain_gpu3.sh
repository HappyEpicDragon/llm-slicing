#!/usr/bin/env bash
set -e
echo '=== E6 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E6_ood.sh
echo '=== E7 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E7_ood.sh
echo '=== E8 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E8_ood.sh
echo '=== E9 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E9_ood.sh
echo '=== E10 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E10_ood.sh
