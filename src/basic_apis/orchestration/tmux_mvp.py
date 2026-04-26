import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


STAGE_ORDER = [
    "native_train",
    "native_test",
    "reevo_search",
    "agentic_eval",
    "summary",
]


def _quote(value) -> str:
    return shlex.quote(str(value))


def _stage_command(cfg, stage_name: str) -> str:
    stage_cfg = cfg.stages[stage_name]
    return str(stage_cfg.command)


def _stage_enabled(cfg, stage_name: str) -> bool:
    return bool(cfg.stages[stage_name].get("enabled", True))


def _stage_env(cfg, stage_name: str) -> str:
    env_cfg = cfg.stages[stage_name].get("env", "")
    return str(env_cfg).strip()


def _tmux_has_session(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _build_pipeline_script(cfg, run_dir: Path) -> str:
    logs_dir = run_dir / "logs"
    status_dir = run_dir / "status"
    reward_path_file = run_dir / "best_reward_path.txt"

    stage_lines = []
    for stage_name in STAGE_ORDER:
        log_path = logs_dir / f"{stage_name}.log"
        status_path = status_dir / f"{stage_name}.status"
        stage_lines.append(f": > {_quote(log_path)}")
        stage_lines.append(f"echo pending > {_quote(status_path)}")

    native_train_cmd = _stage_command(cfg, "native_train")
    native_test_cmd = _stage_command(cfg, "native_test")
    reevo_cmd = _stage_command(cfg, "reevo_search")
    agentic_cmd = _stage_command(cfg, "agentic_eval")
    summary_cmd = _stage_command(cfg, "summary")

    native_train_env = _stage_env(cfg, "native_train")
    native_test_env = _stage_env(cfg, "native_test")
    reevo_env = _stage_env(cfg, "reevo_search")
    agentic_env = _stage_env(cfg, "agentic_eval")
    summary_env = _stage_env(cfg, "summary")

    known_reward = str(cfg.get("reward_fn_path", "")).strip()
    run_native = "true" if _stage_enabled(cfg, "native_train") or _stage_enabled(cfg, "native_test") else "false"
    run_reevo = "true" if _stage_enabled(cfg, "reevo_search") else "false"
    run_agentic = "true" if _stage_enabled(cfg, "agentic_eval") else "false"
    run_summary = "true" if _stage_enabled(cfg, "summary") else "false"

    return f"""#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT={_quote(cfg.project_root)}
RUN_DIR={_quote(run_dir)}
LOGS_DIR={_quote(logs_dir)}
STATUS_DIR={_quote(status_dir)}
BEST_REWARD_FILE={_quote(reward_path_file)}
KNOWN_REWARD_PATH={_quote(known_reward)}

cd "$PROJECT_ROOT"
mkdir -p "$LOGS_DIR" "$STATUS_DIR"
{os.linesep.join(stage_lines)}

mark_status() {{
  local stage="$1"
  local status="$2"
  echo "$status" > "$STATUS_DIR/${{stage}}.status"
  printf '[%s] %s -> %s\\n' "$(date '+%F %T')" "$stage" "$status" >> "$RUN_DIR/timeline.log"
}}

run_stage() {{
  local stage="$1"
  local cmd="$2"
  local env_prefix="$3"
  local log="$LOGS_DIR/${{stage}}.log"
  if [[ "$cmd" == "__disabled__" ]]; then
    mark_status "$stage" "skipped"
    return 0
  fi
  mark_status "$stage" "running"
  set +e
  {{
    echo "===== $stage started at $(date '+%F %T') ====="
    echo "COMMAND: $env_prefix $cmd"
    if [[ -n "$env_prefix" ]]; then
      bash -lc "$env_prefix $cmd"
    else
      bash -lc "$cmd"
    fi
    echo "===== $stage finished at $(date '+%F %T') ====="
  }} > "$log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    mark_status "$stage" "done"
  else
    mark_status "$stage" "failed"
    return "$rc"
  fi
}}

find_best_reward() {{
  if [[ -n "$KNOWN_REWARD_PATH" ]]; then
    echo "$KNOWN_REWARD_PATH" | tee "$BEST_REWARD_FILE"
    return 0
  fi
  python - "$LOGS_DIR/reevo_search.log" "$BEST_REWARD_FILE" <<'PY'
import os
import re
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
text = open(log_path, encoding="utf-8", errors="replace").read()
out_matches = re.findall(r"ReEvo output dir:\\s*(.+)", text)
best_matches = re.findall(r"Evolution complete\\. Best:\\s*(.+)", text)
if not out_matches or not best_matches:
    raise SystemExit("Could not parse ReEvo output dir and best reward path from log")
output_dir = out_matches[-1].strip()
best_path = best_matches[-1].strip()
if not os.path.isabs(best_path):
    best_path = os.path.join(output_dir, best_path)
if not os.path.exists(best_path):
    raise SystemExit(f"Parsed best reward path does not exist: {{best_path}}")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(best_path + "\\n")
print(best_path)
PY
}}

native_pipeline() {{
  if {run_native}; then
    run_stage native_train {_quote(native_train_cmd if _stage_enabled(cfg, "native_train") else "__disabled__")} {_quote(native_train_env)}
    run_stage native_test {_quote(native_test_cmd if _stage_enabled(cfg, "native_test") else "__disabled__")} {_quote(native_test_env)}
  else
    mark_status native_train skipped
    mark_status native_test skipped
  fi
}}

reevo_pipeline() {{
  if {run_reevo}; then
    run_stage reevo_search {_quote(reevo_cmd)} {_quote(reevo_env)}
    find_best_reward
  else
    mark_status reevo_search skipped
    find_best_reward
  fi
}}

agentic_pipeline() {{
  if {run_agentic}; then
    local reward_path
    reward_path="$(cat "$BEST_REWARD_FILE")"
    local cmd={_quote(agentic_cmd)}
    cmd="${{cmd//\\{{reward_fn_path\\}}/$reward_path}}"
    run_stage agentic_eval "$cmd" {_quote(agentic_env)}
  else
    mark_status agentic_eval skipped
  fi
}}

summary_pipeline() {{
  if {run_summary}; then
    local reward_path=""
    if [[ -f "$BEST_REWARD_FILE" ]]; then
      reward_path="$(cat "$BEST_REWARD_FILE")"
    fi
    local cmd={_quote(summary_cmd)}
    MVP_AGENTIC_REWARD_PATH="$reward_path" run_stage summary "$cmd" {_quote(summary_env)}
  else
    mark_status summary skipped
  fi
}}

mark_status pipeline running
native_pipeline &
native_pid=$!
reevo_pipeline &
reevo_pid=$!

wait "$native_pid"
wait "$reevo_pid"
agentic_pipeline
summary_pipeline
mark_status pipeline done
echo "MVP orchestration complete. Run dir: $RUN_DIR"
"""


def _build_monitor_script(cfg, run_dir: Path) -> str:
    logs_dir = run_dir / "logs"
    status_dir = run_dir / "status"
    interval = int(cfg.get("monitor_interval_sec", 10))
    return f"""#!/usr/bin/env bash
set -euo pipefail

RUN_DIR={_quote(run_dir)}
LOGS_DIR={_quote(logs_dir)}
STATUS_DIR={_quote(status_dir)}
INTERVAL={interval}

while true; do
  clear
  echo "Scenario 0 MVP orchestration"
  echo "Run dir: $RUN_DIR"
  echo "Attach: tmux attach -t {_quote(cfg.session_name)}"
  echo
  printf '%-16s %s\\n' "stage" "status"
  printf '%-16s %s\\n' "-----" "------"
  for stage in pipeline native_train native_test reevo_search agentic_eval summary; do
    status_file="$STATUS_DIR/${{stage}}.status"
    if [[ -f "$status_file" ]]; then
      printf '%-16s %s\\n' "$stage" "$(cat "$status_file")"
    else
      printf '%-16s %s\\n' "$stage" "pending"
    fi
  done
  echo
  echo "Timeline:"
  if [[ -f "$RUN_DIR/timeline.log" ]]; then
    tail -n 12 "$RUN_DIR/timeline.log"
  fi
  echo
  echo "Recent logs:"
  for log in "$LOGS_DIR"/*.log; do
    [[ -f "$log" ]] || continue
    echo
    echo "--- $(basename "$log") ---"
    tail -n 8 "$log"
  done
  sleep "$INTERVAL"
done
"""


def run_mvp_agentic_s0(full_cfg) -> dict:
    cfg = full_cfg.orchestration
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required but was not found in PATH")

    session_name = str(cfg.session_name)
    if _tmux_has_session(session_name):
        if bool(cfg.get("force", False)):
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
        else:
            raise RuntimeError(
                f"tmux session '{session_name}' already exists. "
                f"Attach with: tmux attach -t {session_name}, or set orchestration.force=true"
            )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(str(cfg.run_root)).expanduser()
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    OmegaConf.save(full_cfg, run_dir / "resolved_config.yaml", resolve=True)
    pipeline_script = run_dir / "run_pipeline.sh"
    monitor_script = run_dir / "monitor.sh"
    _write(pipeline_script, _build_pipeline_script(cfg, run_dir))
    _write(monitor_script, _build_monitor_script(cfg, run_dir))

    if bool(cfg.get("dry_run", False)):
        print(f"[orchestration] dry run written to: {run_dir}")
        print(f"[orchestration] pipeline: {pipeline_script}")
        print(f"[orchestration] monitor: {monitor_script}")
        return {"run_dir": str(run_dir), "session_name": session_name, "started": False}

    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "-n", "pipeline", "bash"], check=True)
    pipeline_pane = subprocess.check_output(
        ["tmux", "display-message", "-p", "-t", f"{session_name}:pipeline", "#{pane_id}"],
        text=True,
    ).strip()
    subprocess.run(
        ["tmux", "split-window", "-h", "-t", f"{session_name}:pipeline", str(monitor_script)],
        check=True,
    )
    subprocess.run(["tmux", "select-layout", "-t", f"{session_name}:pipeline", "even-horizontal"], check=True)
    subprocess.run(["tmux", "send-keys", "-t", pipeline_pane, str(pipeline_script), "C-m"], check=True)

    print(f"[orchestration] tmux session started: {session_name}")
    print(f"[orchestration] run dir: {run_dir}")
    print(f"[orchestration] attach with: tmux attach -t {session_name}")
    if bool(cfg.get("attach", False)):
        subprocess.run(["tmux", "attach", "-t", session_name], check=False)
    return {"run_dir": str(run_dir), "session_name": session_name, "started": True}
