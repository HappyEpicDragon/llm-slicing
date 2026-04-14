import os
import shutil
from datetime import datetime
from pathlib import Path


def ensure_clean_dir(path: str) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def auto_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_versioned_run_dir(root_dir: str, run_id: str | None = None) -> str:
    rid = run_id or auto_run_id()
    return str(Path(root_dir) / "runs" / rid)


def update_latest_symlink(root_dir: str, run_dir: str) -> str:
    root = Path(root_dir)
    ensure_dir(str(root))
    latest_link = root / "latest"
    rel_target = os.path.relpath(run_dir, root_dir)
    tmp_link = root / ".latest_tmp"

    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(rel_target, tmp_link)
    os.replace(tmp_link, latest_link)
    return str(latest_link)

