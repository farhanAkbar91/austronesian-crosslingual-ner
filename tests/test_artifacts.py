"""Artifact directory creation, manifests, serialization, and bundles."""

import zipfile

import pytest

from axner.artifacts.manager import (
    RUN_SUBDIRS,
    artifact_paths,
    create_run_directory,
    load_json,
    load_jsonl,
    load_manifest,
    load_yaml,
    save_json,
    save_jsonl,
    save_yaml,
    update_manifest,
    write_manifest,
)
from axner.artifacts.transfer import export_bundle, import_bundle
from axner.errors import ArtifactError


def test_create_run_directory_structure(tmp_path):
    run_dir = create_run_directory(tmp_path, "exp0_inlanguage_baseline", label="seed42")
    assert run_dir.parent.name == "exp0_inlanguage_baseline"
    for sub in RUN_SUBDIRS:
        assert (run_dir / sub).is_dir()


def test_run_directories_are_unique(tmp_path):
    first = create_run_directory(tmp_path, "exp1", label="a")
    second = create_run_directory(tmp_path, "exp1", label="a")
    assert first != second


def test_json_round_trip(tmp_path):
    payload = {"nested": {"list": [1, 2, 3], "text": "héllo"}, "number": 1.5}
    save_json(tmp_path / "x" / "a.json", payload)
    assert load_json(tmp_path / "x" / "a.json") == payload


def test_jsonl_round_trip(tmp_path):
    records = [{"a": 1, "tokens": ["wong", "ing"]}, {"a": 2}]
    save_jsonl(records, tmp_path / "x" / "b.jsonl")
    assert load_jsonl(tmp_path / "x" / "b.jsonl") == records


def test_yaml_round_trip(tmp_path):
    payload = {"training": {"epochs": 3, "label_names": ["O", "B-PER"]}}
    save_yaml(payload, tmp_path / "c.yaml")
    assert load_yaml(tmp_path / "c.yaml") == payload


def test_manifest_write_update_load(tmp_path):
    run_dir = create_run_directory(tmp_path, "exp2", label="m")
    manifest = write_manifest(run_dir, experiment_id="exp2", training_seed=42)
    assert manifest["experiment_id"] == "exp2"
    assert manifest["run_id"] == run_dir.name
    updated = update_manifest(run_dir, status="completed", entity_f1=0.42)
    assert updated["status"] == "completed"
    assert updated["training_seed"] == 42
    assert load_manifest(run_dir)["entity_f1"] == 0.42


def test_artifact_paths_layout(tmp_path):
    run_dir = create_run_directory(tmp_path, "exp3", label="p")
    paths = artifact_paths(run_dir)
    assert paths["manifest"] == run_dir / "manifest.json"
    assert paths["sentence_level"] == run_dir / "predictions" / "test_{lang}_sentence_level.jsonl"


def _seed_run(tmp_path, with_checkpoints=False):
    run_dir = create_run_directory(tmp_path, "exp1", label="bundle")
    write_manifest(run_dir, experiment_id="exp1", variant_tags={"source": "id"})
    save_json(run_dir / "metrics" / "test_metrics_jv.json", {"f1": 0.5})
    save_jsonl([{"sentence_id": "s0"}], run_dir / "predictions" / "test_jv_sentence_level.jsonl")
    if with_checkpoints:
        (run_dir / "checkpoints" / "checkpoint-1").mkdir()
        (run_dir / "checkpoints" / "checkpoint-1" / "model.safetensors").write_text("fake")
    return run_dir


def test_bundle_export_import_round_trip(tmp_path):
    run_dir = _seed_run(tmp_path / "src_outputs")
    bundle = export_bundle(run_dir, out_path=tmp_path / "bundles" / "run.zip")
    assert bundle.exists()
    restored = import_bundle(bundle, tmp_path / "dest_outputs")
    assert restored.parent.name == "exp1"
    assert restored.name == run_dir.name
    assert load_manifest(restored)["variant_tags"]["source"] == "id"
    assert load_json(restored / "metrics" / "test_metrics_jv.json") == {"f1": 0.5}


def test_bundle_excludes_checkpoints_by_default(tmp_path):
    run_dir = _seed_run(tmp_path, with_checkpoints=True)
    bundle = export_bundle(run_dir)
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert not any("checkpoints" in name for name in names)


def test_bundle_includes_checkpoints_when_requested(tmp_path):
    run_dir = _seed_run(tmp_path, with_checkpoints=True)
    bundle = export_bundle(run_dir, include_checkpoints=True)
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert any("checkpoints" in name for name in names)


def test_import_refuses_existing_run_without_overwrite(tmp_path):
    run_dir = _seed_run(tmp_path / "src")
    bundle = export_bundle(run_dir)
    import_bundle(bundle, tmp_path / "dst")  # first import is fine
    with pytest.raises(ArtifactError):
        import_bundle(bundle, tmp_path / "dst")
    restored = import_bundle(bundle, tmp_path / "dst", overwrite=True)
    assert restored.name == run_dir.name


def test_import_refuses_non_bundle_zip(tmp_path):
    fake = tmp_path / "fake.zip"
    with zipfile.ZipFile(fake, "w") as zf:
        zf.writestr("random.txt", "not a bundle")
    with pytest.raises(ArtifactError):
        import_bundle(fake, tmp_path / "outputs")


def test_export_refuses_run_without_manifest(tmp_path):
    bare = tmp_path / "not_a_run"
    bare.mkdir()
    with pytest.raises(ArtifactError):
        export_bundle(bare)
