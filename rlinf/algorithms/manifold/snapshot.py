# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integrity and loaded-module evidence for an explicit training snapshot."""

import hashlib
import json
import os
import socket
import sys
import uuid
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    """Describe a regular file or an internal relative symlink."""
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    return {
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "executable": bool(path.stat().st_mode & 0o111),
    }


def source_files(root: Path) -> dict:
    """Inventory a frozen source tree, excluding Python bytecode caches."""
    return {
        str(path.relative_to(root)): file_identity(path)
        for path in sorted(root.rglob("*"))
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    }


def verify_snapshot(run: Path) -> dict:
    """Reject changed, missing, or extra files in a prepared source/config."""
    manifest = json.loads((run / "manifests/source.json").read_text())
    root = run / "source"
    actual = source_files(root)
    if actual != manifest["files"]:
        changed = sorted(
            key
            for key in actual.keys() | manifest["files"].keys()
            if actual.get(key) != manifest["files"].get(key)
        )
        raise RuntimeError(f"Frozen mani_v1 source changed: {changed[:20]}")
    encoded = json.dumps(actual, sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["tree_sha256"]:
        raise RuntimeError("Expected a matching mani_v1 source tree digest.")
    if sha256(run / "configs/train.yaml") != manifest["config_sha256"]:
        raise RuntimeError("Frozen mani_v1 resolved config changed.")
    return manifest


def verify_runtime_source(role: str) -> dict | None:
    """Check loaded RLinf modules and write one process/role provenance report.

    The snapshot launcher sets RLINF_MANIFOLD_RUN. Ordinary launches do no I/O.
    """
    selected = os.environ.get("RLINF_MANIFOLD_RUN")
    if selected is None:
        return None
    run = Path(selected).resolve()
    manifest = verify_snapshot(run)
    root = (run / "source").resolve()
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name != "rlinf" and not name.startswith("rlinf."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(
                f"Expected {name} from frozen source {root}, found {path}."
            )
        key = str(path.relative_to(root))
        expected = manifest["files"].get(key, {})
        actual = sha256(path)
        if expected.get("sha256") != actual:
            raise RuntimeError(f"Loaded mani_v1 module hash differs: {name}: {path}")
        loaded[name] = {"file": str(path), "sha256": actual}
    report = {
        "role": role,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "source_tree_sha256": manifest["tree_sha256"],
        "modules": loaded,
    }
    directory = run / "runtime/source_reports"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{role}_{os.getpid()}_{uuid.uuid4().hex}.json"
    with destination.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return report
