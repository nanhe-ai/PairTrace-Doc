import hashlib
from pathlib import Path

import yaml

from pairtrace_doc.pipelines.acquire_descan18k_test import run


class _Response:
    def __init__(self, *, status_code: int, headers=None, payload: bytes = b"") -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_acquire_descan18k_downloads_without_reading_or_decoding(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path
    config_dir = project_root / "configs"
    config_dir.mkdir()
    approval = project_root / "docs" / "approval.md"
    approval.parent.mkdir()
    approval.write_text("approved\n", encoding="utf-8")
    payload = b"not-opened-archive-payload"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    revision = "a" * 40
    config = {
        "experiment": {"name": "fixture", "paper_evidence": False},
        "source": {
            "repository": "fixture",
            "repository_revision": revision,
            "archive_url": "https://example.invalid/Test.zip",
            "expected_archive_bytes": len(payload),
            "expected_archive_sha256": payload_sha256,
        },
        "license": {
            "approval_record": "docs/approval.md",
            "expected_approval_record_sha256": hashlib.sha256(
                approval.read_bytes()
            ).hexdigest(),
        },
        "runtime": {
            "license_gate_open": True,
            "archive_download_authorized": True,
            "archive_read_authorized": False,
            "dataset_image_decode_authorized": False,
            "model_scoring_authorized": False,
        },
        "paths": {
            "archive": "data/Test.zip",
            "summary": "outputs/acquisition.json",
        },
    }
    config_path = config_dir / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        "pairtrace_doc.pipelines.acquire_descan18k_test.requests.head",
        lambda *_args, **_kwargs: _Response(
            status_code=302,
            headers={
                "X-Repo-Commit": revision,
                "X-Linked-Size": str(len(payload)),
                "X-Linked-ETag": f'"{payload_sha256}"',
            },
        ),
    )
    monkeypatch.setattr(
        "pairtrace_doc.pipelines.acquire_descan18k_test.requests.get",
        lambda *_args, **_kwargs: _Response(status_code=200, payload=payload),
    )

    summary = run(config_path)

    assert summary["status"] == "descan18k_test_archive_acquired_verified"
    assert summary["archive_read"] is False
    assert summary["dataset_image_decoded"] is False
    assert (project_root / "data" / "Test.zip").read_bytes() == payload
