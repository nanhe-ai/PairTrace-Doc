from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


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
        return {"status": "passed", "waived_issues": [], "output": "\n".join(lines)}
    waived = [
        line for line in lines if any(line.startswith(prefix) for prefix in waiver_prefixes)
    ]
    unwaived = [line for line in lines if line not in waived]
    if not lines or unwaived:
        diagnostic = " | ".join(unwaived or ["pip check returned no diagnostics"])
        raise ValueError(f"pip check found unwaived issues: {diagnostic}")
    return {
        "status": "passed_with_declared_packaging_metadata_waivers",
        "waived_issues": waived,
        "output": "\n".join(lines),
    }


def _environment_state(
    project_root: Path,
    editor: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    environment = _resolve(project_root, str(editor["environment_path"]))
    python = environment / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"environment Python is missing: {python}")
    metadata_environment = os.environ.copy()
    metadata_environment.pop("PYTHONPATH", None)

    version_process = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata,json,sys; "
                "print(json.dumps({x:importlib.metadata.version(x) for x in sys.argv[1:]}))"
            ),
            "torch",
            *( ["torchvision"] if editor.get("torchvision") else [] ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=metadata_environment,
    )
    versions = json.loads(version_process.stdout)
    expected_versions = {"torch": str(editor["torch"])}
    if editor.get("torchvision"):
        expected_versions["torchvision"] = str(editor["torchvision"])
    if versions != expected_versions:
        raise ValueError(
            f"runtime version mismatch for {editor['id']}: {versions} != {expected_versions}"
        )

    pip_check_process = subprocess.run(
        [str(python), "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=metadata_environment,
    )
    pip_check = _classify_pip_check(
        pip_check_process.returncode,
        pip_check_process.stdout,
        pip_check_process.stderr,
        [str(value) for value in editor.get("pip_check_waiver_prefixes", [])],
    )

    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--disable-pip-version-check"],
        check=True,
        capture_output=True,
        text=True,
        env=metadata_environment,
    )
    lock_lines = sorted(
        (line.strip() for line in freeze.stdout.splitlines() if line.strip()),
        key=str.casefold,
    )
    lock_path = _resolve(project_root, str(editor["resolved_lock"]))
    _write_text(lock_path, "\n".join(lock_lines) + "\n")

    probe_environment = os.environ.copy()
    probe_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    code_paths = [
        _resolve(project_root, str(value)) for value in editor.get("code_paths", [])
    ]
    if code_paths:
        probe_environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in code_paths)
    probe_code = (
        "import importlib,json,sys,torch; "
        "[importlib.import_module(x) for x in sys.argv[1:]]; "
        "x=torch.ones(%d,device='cuda'); "
        "torch.cuda.synchronize(); "
        "print(json.dumps({"
        "'torch':torch.__version__,"
        "'device_name':torch.cuda.get_device_name(0),"
        "'compute_capability':list(torch.cuda.get_device_capability(0)),"
        "'arch_list':torch.cuda.get_arch_list(),"
        "'tensor_sum':float(x.sum().item()),"
        "'allocated_bytes':torch.cuda.memory_allocated(0)" 
        "}))"
    ) % int(validation["synthetic_cuda_elements"])
    probe = subprocess.run(
        [
            str(python),
            "-c",
            probe_code,
            *[str(value) for value in editor["import_modules"]],
        ],
        check=True,
        capture_output=True,
        text=True,
        env=probe_environment,
    )
    device = json.loads(probe.stdout.splitlines()[-1])
    if device["device_name"] != str(validation["expected_gpu_name"]):
        raise ValueError(f"unexpected GPU for {editor['id']}: {device['device_name']}")
    if device["compute_capability"] != list(validation["expected_compute_capability"]):
        raise ValueError(
            f"unexpected compute capability for {editor['id']}: {device['compute_capability']}"
        )
    if str(validation["expected_architecture"]) not in device["arch_list"]:
        raise ValueError(f"sm_120 is missing for {editor['id']}")
    if device["tensor_sum"] != float(validation["synthetic_cuda_elements"]):
        raise ValueError(f"synthetic CUDA kernel returned the wrong value for {editor['id']}")

    return {
        "editor_id": str(editor["id"]),
        "environment_path": str(environment),
        "versions": versions,
        "pip_check": pip_check,
        "lock_path": str(lock_path),
        "lock_sha256": _sha256(lock_path),
        "lock_packages": len(lock_lines),
        "imports": [str(value) for value in editor["import_modules"]],
        "device_probe": device,
        "probe_stderr": probe.stderr.strip(),
        "status": "passed_model_not_loaded",
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    for field, expected in config["frozen_bindings"].items():
        if not field.endswith("_sha256"):
            continue
        path = _resolve(
            project_root, str(config["frozen_bindings"][field.removesuffix("_sha256")])
        )
        if _sha256(path) != str(expected):
            raise ValueError(f"frozen binding changed: {path}")

    storage = shutil.disk_usage(_resolve(project_root, "../autodl-tmp/pairtrace-doc"))
    floor = int(float(config["validation"]["minimum_free_space_gib"]) * 1024**3)
    if storage.free < floor:
        raise ValueError(f"persistent storage floor violated: {storage.free} < {floor}")

    states = [
        _environment_state(project_root, editor, config["validation"])
        for editor in config["editors"]
    ]
    result = {
        "authorization": {
            "model_loaded": False,
            "editor_inference_run": False,
            "detector_inference_run": False,
            "nonfinal_toy_images_read": 0,
            "final_source_images_read": 0,
        },
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "editors": states,
        "configured_editors": len(states),
        "storage": {
            "free_bytes": storage.free,
            "minimum_free_bytes": floor,
        },
        "status": (
            "passed_all_four_sm120_runtime_model_not_loaded"
            if len(states) == 4
            else "passed_all_configured_sm120_runtimes_model_not_loaded"
        ),
    }
    report_path = _resolve(project_root, str(config["outputs"]["report"]))
    _write_text(
        report_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    result["report_path"] = str(report_path)
    result["report_sha256"] = _sha256(report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact SM120-compatible editor runtimes without loading models."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.config, args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
