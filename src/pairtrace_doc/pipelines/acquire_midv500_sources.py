from __future__ import annotations

import argparse
import ftplib
import hashlib
import io
import json
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class _FTPRangeReader(io.RawIOBase):
    """Seekable, read-only FTP view used for selective ZIP extraction."""

    def __init__(self, host: str, remote_path: str, timeout_seconds: int) -> None:
        self.host = host
        self.remote_path = remote_path
        self.timeout_seconds = timeout_seconds
        self.position = 0
        with ftplib.FTP(host, timeout=timeout_seconds) as ftp:
            ftp.login()
            ftp.voidcmd("TYPE I")
            size = ftp.size(remote_path)
        if size is None:
            raise RuntimeError(f"FTP server returned no size for {remote_path}")
        self.size = int(size)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""

        ftp = ftplib.FTP(self.host, timeout=self.timeout_seconds)
        ftp.login()
        ftp.voidcmd("TYPE I")
        data_socket = ftp.transfercmd(
            f"RETR {self.remote_path}", rest=self.position
        )
        chunks: list[bytes] = []
        remaining = size
        try:
            while remaining:
                chunk = data_socket.recv(min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            data_socket.close()
            ftp.close()
        payload = b"".join(chunks)
        self.position += len(payload)
        if len(payload) != size:
            raise EOFError(
                f"short FTP range read for {self.remote_path}: "
                f"expected {size}, received {len(payload)}"
            )
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ftp_names(host: str, remote_directory: str, timeout_seconds: int) -> list[str]:
    with ftplib.FTP(host, timeout=timeout_seconds) as ftp:
        ftp.login()
        ftp.cwd(remote_directory)
        return sorted(ftp.nlst())


def _ftp_download(
    host: str, remote_path: str, destination: Path, timeout_seconds: int
) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with ftplib.FTP(host, timeout=timeout_seconds) as ftp:
        ftp.login()
        ftp.voidcmd("TYPE I")
        with temporary.open("wb") as handle:
            ftp.retrbinary(f"RETR {remote_path}", handle.write, blocksize=1024 * 1024)
    temporary.replace(destination)


def _copy_member(
    archive: zipfile.ZipFile, member: str, destination: Path
) -> dict[str, Any]:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with archive.open(info, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary.replace(destination)
    if destination.stat().st_size != info.file_size:
        raise ValueError(
            f"size mismatch for {destination}: "
            f"{destination.stat().st_size} != {info.file_size}"
        )
    return {
        "member": member,
        "bytes": destination.stat().st_size,
        "crc32": f"{info.CRC:08x}",
        "sha256": _sha256(destination),
        "path": destination.as_posix(),
    }


def acquire(config_path: Path, project_root: Path, storage_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    midv = config["datasets"]["midv500"]
    expected = int(midv["expected_document_types"])
    pattern = re.compile(str(midv["archive_pattern"]))
    ftp_root = str(midv["official_ftp_root"])
    match = re.fullmatch(r"ftp://([^/]+)(/.*)", ftp_root)
    if match is None:
        raise ValueError("official_ftp_root must be an ftp:// URL")
    host = match.group(1)
    root_path = match.group(2).rstrip("/")
    dataset_root = str(midv["official_dataset_root"]).rstrip("/")
    timeout_seconds = 120

    names = _ftp_names(host, dataset_root, timeout_seconds)
    archives = [name for name in names if pattern.fullmatch(Path(name).name)]
    if len(archives) != expected:
        raise ValueError(f"expected {expected} archives, found {len(archives)}")

    dataset_dir = storage_root / config["storage"]["relative_root"] / "midv500"
    provenance_dir = dataset_dir / "provenance"
    provenance: list[dict[str, Any]] = []
    for name in midv["provenance_allowlist"]:
        destination = provenance_dir / name
        _ftp_download(host, f"{root_path}/{name}", destination, timeout_seconds)
        provenance.append(
            {
                "name": name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "path": destination.as_posix(),
            }
        )

    def acquire_archive(item: tuple[int, str]) -> dict[str, Any]:
        index, archive_name = item
        stem = Path(archive_name).stem
        remote_path = f"{dataset_root}/{archive_name}"
        remote = _FTPRangeReader(host, remote_path, timeout_seconds)
        members: list[dict[str, Any]] = []
        with zipfile.ZipFile(remote) as archive:
            for template in midv["selective_members"]:
                member = str(template).format(stem=stem)
                destination = dataset_dir / "sources" / stem / Path(member).name
                members.append(_copy_member(archive, member, destination))
        print(f"[{index:02d}/{expected}] {stem}", flush=True)
        return {
            "document_type_index": index,
            "document_type": stem,
            "remote_archive": f"ftp://{host}{remote_path}",
            "remote_archive_bytes": remote.size,
            "selected_members": members,
            "video_frames_selected": 0,
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        rows = list(executor.map(acquire_archive, enumerate(archives, start=1)))

    output_path = project_root / config["paths"]["midv500_manifest"]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_path.relative_to(project_root).as_posix(),
        "config_sha256": _sha256(config_path),
        "official_ftp_root": ftp_root,
        "license": midv["license"],
        "expected_document_types": expected,
        "acquired_document_types": len(rows),
        "video_frames_selected": 0,
        "provenance": provenance,
        "sources": rows,
    }
    _write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selectively acquire the 50 MIDV-500 source images."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage-root", required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    result = acquire(config_path, project_root, Path(args.storage_root))
    print(
        json.dumps(
            {
                "acquired_document_types": result["acquired_document_types"],
                "video_frames_selected": result["video_frames_selected"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
