from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests
import yaml


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    if not bool(runtime["license_gate_open"]) or not bool(
        runtime["archive_download_authorized"]
    ):
        raise PermissionError("DESCAN archive download gate is closed")
    if any(
        bool(runtime[name])
        for name in (
            "archive_read_authorized",
            "dataset_image_decode_authorized",
            "model_scoring_authorized",
        )
    ):
        raise ValueError("archive acquisition cannot authorize read, decode, or scoring")

    approval = _resolve(project_root, str(config["license"]["approval_record"]))
    expected_approval_sha256 = str(
        config["license"]["expected_approval_record_sha256"]
    )
    if _sha256(approval) != expected_approval_sha256:
        raise ValueError("DESCAN approval record changed")

    source = config["source"]
    expected_bytes = int(source["expected_archive_bytes"])
    expected_sha256 = str(source["expected_archive_sha256"])
    revision = str(source["repository_revision"])
    url = str(source["archive_url"])
    archive = _resolve(project_root, str(config["paths"]["archive"]))
    summary_path = _resolve(project_root, str(config["paths"]["summary"]))
    archive.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "descan18k_test_acquisition_failed",
        "paper_evidence": False,
        "archive_read": False,
        "dataset_image_decoded": False,
        "model_scoring_started": False,
        "source_url": url,
        "repository_revision": revision,
        "expected_archive_bytes": expected_bytes,
        "expected_archive_sha256": expected_sha256,
        "approval_record": str(approval.relative_to(project_root)),
        "approval_record_sha256": expected_approval_sha256,
        "config_sha256": _sha256(config_path),
        "archive": str(archive),
        "cache_key": f"descan18k:{revision}:Test.zip:{expected_sha256}",
        "failure_reason": None,
    }
    try:
        cache_hit = archive.is_file()
        if not cache_hit:
            head = requests.head(url, allow_redirects=False, timeout=30)
            head.raise_for_status()
            observed_revision = head.headers.get("X-Repo-Commit")
            observed_bytes = int(head.headers.get("X-Linked-Size", "-1"))
            observed_sha256 = head.headers.get("X-Linked-ETag", "").strip('"')
            if observed_revision != revision:
                raise ValueError("DESCAN repository revision changed")
            if observed_bytes != expected_bytes:
                raise ValueError("DESCAN remote byte count changed")
            if observed_sha256 != expected_sha256:
                raise ValueError("DESCAN remote linked SHA-256 changed")

            partial = archive.with_suffix(archive.suffix + ".part")
            offset = partial.stat().st_size if partial.is_file() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(
                url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("DESCAN server did not honor resumable range request")
                mode = "ab" if offset else "wb"
                downloaded = offset
                next_report = ((downloaded // (64 * 1024 * 1024)) + 1) * (
                    64 * 1024 * 1024
                )
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                json.dumps(
                                    {
                                        "event": "download_progress",
                                        "bytes": downloaded,
                                        "expected_bytes": expected_bytes,
                                    }
                                ),
                                flush=True,
                            )
                            next_report += 64 * 1024 * 1024
            if partial.stat().st_size != expected_bytes:
                raise ValueError("DESCAN downloaded byte count changed")
            if _sha256(partial) != expected_sha256:
                raise ValueError("DESCAN downloaded SHA-256 changed")
            partial.replace(archive)

        if archive.stat().st_size != expected_bytes:
            raise ValueError("DESCAN cached archive byte count changed")
        actual_sha256 = _sha256(archive)
        if actual_sha256 != expected_sha256:
            raise ValueError("DESCAN cached archive SHA-256 changed")
        summary.update(
            {
                "status": "descan18k_test_archive_acquired_verified",
                "cache_hit": cache_hit,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": actual_sha256,
            }
        )
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        _write_json(summary_path, summary)
        raise
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
