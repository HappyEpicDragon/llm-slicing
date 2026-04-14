#!/usr/bin/env bash
set -e
echo '=== E1 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E1_ood.sh
echo '=== E2 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E2_ood.sh
echo '=== E3 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E3_ood.sh
echo '=== E4 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E4_ood.sh
echo '=== E5 ==='
bash /root/decision_transformer_slicing/logs/dt_eval_8exp/scripts/eval_E5_ood.sh
