"""Freeze the final pipeline before test evaluation, and verify it has not changed since."""
import hashlib
import json
import subprocess
from pathlib import Path

from glaucoma_project import REPO_ROOT

MANIFEST = REPO_ROOT / "docs" / "frozen_baseline_a.json"
CODE_PATHS = ["src", "configs"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def freeze(files: dict[str, Path], settings: dict, manifest: Path = MANIFEST) -> dict:
    """Record the code version, the hash of each frozen file and the chosen settings. Never overwrites."""
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists: the pipeline is already frozen")
    if _git("status", "--porcelain", "--", *CODE_PATHS):
        raise RuntimeError("Commit src/ and configs/ before freezing")
    record = {"git_commit": _git("rev-parse", "HEAD"),
              "files": {name: {"path": str(Path(p).relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
                        for name, p in files.items()},
              "settings": settings}
    manifest.write_text(json.dumps(record, indent=2))
    return record


def verify(manifest: Path = MANIFEST) -> dict:
    """Raise unless code and frozen files are exactly as they were when frozen. Returns the manifest."""
    record = json.loads(manifest.read_text())
    if subprocess.run(["git", "diff", "--quiet", record["git_commit"], "--", *CODE_PATHS], cwd=REPO_ROOT).returncode:
        raise RuntimeError("src/ or configs/ changed since the pipeline was frozen")
    for name, entry in record["files"].items():
        if sha256_file(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")
    return record
