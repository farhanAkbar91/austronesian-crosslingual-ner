"""Run artifact bundles for Colab/Kaggle <-> local transfer.

A bundle is a zip of one run directory (optionally without checkpoints to
keep downloads small). Importing verifies the manifest and restores the run
under ``<outputs_root>/<experiment_id>/<run_id>/`` so locally produced and
cloud-produced runs are interchangeable.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from axner.artifacts.manager import load_json
from axner.errors import ArtifactError


def export_bundle(run_dir, out_path=None, include_checkpoints: bool = False) -> Path:
    """Zip one run directory into a transferable bundle."""
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        raise ArtifactError(f"refusing to bundle {run_dir}: manifest.json missing")
    if out_path is None:
        out_path = run_dir.parent / f"{run_dir.name}.bundle.zip"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            if not include_checkpoints and "checkpoints" in path.relative_to(run_dir).parts:
                continue
            zf.write(path, path.relative_to(run_dir))
    return out_path


def import_bundle(bundle_path, outputs_root, overwrite: bool = False) -> Path:
    """Extract a bundle into the outputs tree; returns the restored run directory."""
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise ArtifactError(f"bundle not found: {bundle_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            with zipfile.ZipFile(bundle_path) as zf:
                zf.extractall(tmp_dir)
        except zipfile.BadZipFile as exc:
            raise ArtifactError(f"invalid bundle zip: {bundle_path}") from exc

        manifest_path = tmp_dir / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactError(
                f"bundle {bundle_path.name} has no manifest.json at its root; "
                "it may not be a run bundle"
            )
        manifest = load_json(manifest_path)
        experiment_id = manifest.get("experiment_id")
        run_id = manifest.get("run_id")
        if not experiment_id or not run_id:
            raise ArtifactError("bundle manifest is missing experiment_id/run_id")

        destination = Path(outputs_root) / experiment_id / run_id
        if destination.exists():
            if not overwrite:
                raise ArtifactError(
                    f"run directory already exists: {destination} (use overwrite=True)"
                )
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_dir), str(destination))
    return destination
