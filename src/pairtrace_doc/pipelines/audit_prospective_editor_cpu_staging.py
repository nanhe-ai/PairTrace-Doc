from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


_DOWNLOAD_CACHE_PARTS = frozenset({".cache", ".huggingface"})


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _write_text(path, payload)


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"model directory is missing: {root}")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if path.name.endswith(".incomplete"):
            raise ValueError(f"incomplete model download remains: {path}")
        if _DOWNLOAD_CACHE_PARTS.intersection(relative.parts):
            continue
        size = path.stat().st_size
        file_sha256 = _sha256(path)
        relative_text = relative.as_posix()
        total_bytes += size
        aggregate.update(f"{file_sha256}  {relative_text}\n".encode("utf-8"))
        rows.append(
            {
                "bytes": size,
                "path": relative_text,
                "sha256": file_sha256,
            }
        )
    return rows, total_bytes, aggregate.hexdigest()


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _classify_pip_check(
    returncode: int,
    stdout: str,
    stderr: str,
    waiver_prefixes: list[str],
) -> dict[str, Any]:
    lines = [
        line.strip()
        for line in (*stdout.splitlines(), *stderr.splitlines())
        if line.strip()
    ]
    if returncode == 0:
        return {
            "output": "\n".join(lines) or "No broken requirements found.",
            "status": "passed",
            "waived_issues": [],
            "waiver_prefixes": waiver_prefixes,
        }

    waived = [
        line for line in lines if any(line.startswith(prefix) for prefix in waiver_prefixes)
    ]
    unwaived = [line for line in lines if line not in waived]
    if not lines:
        raise ValueError(f"pip check failed without diagnostic output: {returncode}")
    if unwaived:
        raise ValueError("pip check found unwaived issues: " + " | ".join(unwaived))
    return {
        "output": "\n".join(lines),
        "status": "passed_with_declared_packaging_metadata_waivers",
        "waived_issues": waived,
        "waiver_prefixes": waiver_prefixes,
    }


def _git_state(path: Path, expected_revision: str) -> dict[str, Any]:
    actual = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if actual != expected_revision:
        raise ValueError(f"code revision mismatch at {path}: {actual}")
    if status:
        raise ValueError(f"code checkout is dirty: {path}")
    return {"path": str(path), "revision": actual, "clean_checkout": True}


def _environment_state(
    environment: Path,
    expected_python: str,
    expected_package_versions: dict[str, str],
    required_lock_substrings: list[str],
    import_modules: list[str],
    deferred_gpu_import_modules: list[dict[str, str]],
    lock_path: Path,
    code_paths: list[Path],
    pip_check_waiver_prefixes: list[str],
) -> dict[str, Any]:
    python = environment / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"environment Python is missing: {python}")
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import json,sys; print(json.dumps(list(sys.version_info[:3])))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    version = json.loads(probe.stdout)
    major_minor = f"{version[0]}.{version[1]}"
    if major_minor != expected_python:
        raise ValueError(f"Python mismatch in {environment}: {major_minor}")

    version_probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata,json,sys; "
                "print(json.dumps({x:importlib.metadata.version(x) for x in sys.argv[1:]}))"
            ),
            *expected_package_versions,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    installed_versions = json.loads(version_probe.stdout)
    if installed_versions != expected_package_versions:
        raise ValueError(
            f"package version mismatch in {environment}: "
            f"{installed_versions} != {expected_package_versions}"
        )

    pip_check_process = subprocess.run(
        [str(python), "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    pip_check = _classify_pip_check(
        pip_check_process.returncode,
        pip_check_process.stdout,
        pip_check_process.stderr,
        pip_check_waiver_prefixes,
    )
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
    )
    lock_lines = sorted(
        (line.strip() for line in freeze.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )
    lock_text = "\n".join(lock_lines) + "\n"
    for required in required_lock_substrings:
        if required not in lock_text:
            raise ValueError(f"environment lock is missing frozen identity: {required}")
    _write_text(lock_path, lock_text)

    smoke_environment = os.environ.copy()
    smoke_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if code_paths:
        existing_pythonpath = smoke_environment.get("PYTHONPATH")
        pieces = [str(path) for path in code_paths]
        if existing_pythonpath:
            pieces.append(existing_pythonpath)
        smoke_environment["PYTHONPATH"] = os.pathsep.join(pieces)
    smoke = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib,sys; [importlib.import_module(x) for x in sys.argv[1:]]",
            *import_modules,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=smoke_environment,
    )
    return {
        "bytes": _directory_bytes(environment),
        "cpu_import_modules": import_modules,
        "cpu_import_smoke_test": "passed_with_CUDA_VISIBLE_DEVICES_empty",
        "deferred_gpu_import_modules": deferred_gpu_import_modules,
        "lock_packages": len(lock_lines),
        "path": str(environment),
        "pip_check": pip_check["output"],
        "pip_check_status": pip_check["status"],
        "pip_check_waived_issues": pip_check["waived_issues"],
        "pip_check_waiver_prefixes": pip_check["waiver_prefixes"],
        "required_package_versions": installed_versions,
        "required_lock_substrings": required_lock_substrings,
        "python": major_minor,
        "resolved_lock": str(lock_path),
        "resolved_lock_sha256": _sha256(lock_path),
        "smoke_stderr": smoke.stderr.strip(),
    }


def run(config_path: Path, project_root: Path, editor_id: str) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    freeze_path = _resolve(project_root, str(config["frozen_editor_config"]))
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    frozen_editors = {str(row["id"]): row for row in freeze["editors"]}
    staging_editors = {str(row["id"]): row for row in config["editors"]}
    if editor_id not in staging_editors or editor_id not in frozen_editors:
        raise ValueError(f"unknown editor ID: {editor_id}")
    staging = staging_editors[editor_id]
    frozen = frozen_editors[editor_id]

    for field in ("model_repository", "model_revision"):
        if str(staging[field]) != str(frozen[field]):
            raise ValueError(f"{field} differs from frozen editor protocol")

    code_states = []
    code_paths = []
    for checkout in staging["code_checkouts"]:
        field = str(checkout["frozen_revision_field"])
        expected_revision = str(frozen[field])
        if str(checkout["revision"]) != expected_revision:
            raise ValueError(f"checkout revision differs from frozen {field}")
        code_path = _resolve(project_root, str(checkout["path"]))
        code_states.append(_git_state(code_path, expected_revision))
        code_paths.append(code_path)

    model_root = _resolve(project_root, str(staging["model_path"]))
    inventory, model_bytes, aggregate_sha256 = _inventory(model_root)
    expected_files = int(staging["expected_model_files"])
    expected_bytes = int(frozen["current_repository_bytes_at_freeze"])
    if len(inventory) != expected_files:
        raise ValueError(
            f"model file count mismatch for {editor_id}: "
            f"{len(inventory)} != {expected_files}"
        )
    if model_bytes != expected_bytes:
        raise ValueError(
            f"model byte total mismatch for {editor_id}: "
            f"{model_bytes} != {expected_bytes}"
        )
    inventory_path = _resolve(project_root, str(staging["inventory_path"]))
    _write_jsonl(inventory_path, inventory)

    lock_path = _resolve(project_root, str(staging["environment_lock_path"]))
    environment = _environment_state(
        environment=_resolve(project_root, str(staging["environment_path"])),
        expected_python=str(frozen["python"]),
        expected_package_versions={
            str(key): str(value)
            for key, value in staging["expected_package_versions"].items()
        },
        required_lock_substrings=[
            str(value) for value in staging.get("required_lock_substrings", [])
        ],
        import_modules=[str(value) for value in staging["cpu_import_modules"]],
        deferred_gpu_import_modules=[
            {
                "module": str(value["module"]),
                "reason": str(value["reason"]),
                "status": "deferred_until_gpu_mount_no_model_load",
            }
            for value in staging.get("deferred_gpu_import_modules", [])
        ],
        lock_path=lock_path,
        code_paths=code_paths,
        pip_check_waiver_prefixes=[
            str(value) for value in staging.get("pip_check_waiver_prefixes", [])
        ],
    )

    storage_root = _resolve(project_root, str(config["storage_root"]))
    storage = shutil.disk_usage(storage_root)
    floor = int(float(config["minimum_free_space_gib"]) * 1024**3)
    if storage.free < floor:
        raise ValueError(f"persistent storage floor violated: {storage.free} < {floor}")

    result = {
        "authorization": {
            "detector_inference_run": False,
            "editor_inference_run": False,
            "final_source_images_read": 0,
            "model_loaded": False,
            "source_manifests_read": False,
        },
        "code": code_states,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "editor_id": editor_id,
        "environment": environment,
        "frozen_editor_config": str(freeze_path),
        "frozen_editor_config_sha256": _sha256(freeze_path),
        "model": {
            "aggregate_inventory_sha256": aggregate_sha256,
            "aggregate_inventory_sha256_definition": (
                "SHA-256 of sorted relative-path-prefixed sha256sum lines, "
                "excluding downloader cache directories"
            ),
            "bytes_excluding_download_cache": model_bytes,
            "files_excluding_download_cache": len(inventory),
            "inventory": str(inventory_path),
            "inventory_sha256": _sha256(inventory_path),
            "path": str(model_root),
            "repository": str(frozen["model_repository"]),
            "revision": str(frozen["model_revision"]),
        },
        "status": "cpu_staging_passed_model_not_loaded",
        "storage": {
            "available_bytes": storage.free,
            "minimum_free_bytes": floor,
            "root": str(storage_root),
        },
    }
    report_path = _resolve(project_root, str(staging["report_path"]))
    _write_json(report_path, result)
    result["report_path"] = str(report_path)
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one exact prospective editor after CPU-only staging without "
            "loading its model or reading a source manifest."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--editor-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config, args.project_root, args.editor_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
