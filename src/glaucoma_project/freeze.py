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


def freeze(files: dict[str, Path], settings: dict, manifest: Path = MANIFEST,
           code_paths: list[str] = CODE_PATHS) -> dict:
    """Record the code version, the hash of each frozen file and the chosen settings. Never overwrites.
    ``code_paths`` are the folders and files whose later change invalidates the freeze (default: all
    of src and configs); they are stored, so ``verify`` checks the same ones."""
    if manifest.exists():
        raise FileExistsError(f"{manifest} already exists: the pipeline is already frozen")
    if _git("status", "--porcelain", "--", *code_paths):
        raise RuntimeError("Commit the frozen code and configs before freezing")
    record = {"git_commit": _git("rev-parse", "HEAD"), "code_paths": list(code_paths),
              "files": {name: {"path": str(Path(p).relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
                        for name, p in files.items()},
              "settings": settings}
    manifest.write_text(json.dumps(record, indent=2))
    return record


def verify(manifest: Path = MANIFEST) -> dict:
    """Raise unless code and frozen files are exactly as they were when frozen. Returns the manifest."""
    record = json.loads(manifest.read_text())
    paths = record.get("code_paths", CODE_PATHS)
    if subprocess.run(["git", "diff", "--quiet", record["git_commit"], "--", *paths], cwd=REPO_ROOT).returncode:
        raise RuntimeError("Frozen code or configs changed since the freeze")
    return verify_files(manifest)


def verify_files(manifest: Path = MANIFEST) -> dict:
    """Raise unless every frozen file still has its recorded hash, ignoring the code. For reusing
    a frozen artefact (such as the segmentation weights) from a later experiment. Returns the manifest."""
    record = json.loads(manifest.read_text())
    for name, entry in record["files"].items():
        if sha256_file(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")
    return record
