from __future__ import annotations

import argparse
import binascii
import codecs
import hashlib
import io
import json
import struct
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml


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


def _request_range(
    url: str, start: int, end: int, timeout_seconds: int = 180
) -> bytes:
    for attempt in range(3):
        response = requests.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=timeout_seconds,
        )
        if response.status_code == 206 and len(response.content) == end - start + 1:
            return response.content
        if attempt == 2:
            raise RuntimeError(
                f"range {start}-{end} failed: status={response.status_code}, "
                f"bytes={len(response.content)}"
            )
    raise AssertionError("unreachable")


def _parallel_range(url: str, start: int, length: int, workers: int = 8) -> bytes:
    if length <= 0:
        return b""
    if length < 1024 * 1024 or workers == 1:
        return _request_range(url, start, start + length - 1)
    parts = min(workers, max(1, (length + 1024 * 1024 - 1) // (1024 * 1024)))
    boundaries = [length * index // parts for index in range(parts + 1)]
    spans = [
        (start + boundaries[index], start + boundaries[index + 1] - 1)
        for index in range(parts)
    ]
    with ThreadPoolExecutor(max_workers=parts) as executor:
        chunks = list(executor.map(lambda span: _request_range(url, *span), spans))
    payload = b"".join(chunks)
    if len(payload) != length:
        raise RuntimeError(f"parallel range returned {len(payload)} != {length}")
    return payload


class _HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, expected_bytes: int) -> None:
        self.url = url
        self.size = expected_bytes
        self.position = 0
        response = requests.head(url, timeout=60)
        response.raise_for_status()
        observed = int(response.headers["Content-Length"])
        if observed != expected_bytes:
            raise ValueError(f"remote size changed: {observed} != {expected_bytes}")

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
        payload = _parallel_range(self.url, self.position, size)
        self.position += len(payload)
        return payload


def _member_bytes(url: str, info: zipfile.ZipInfo) -> bytes:
    header = _request_range(url, info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"invalid local ZIP header for {info.filename}")
    filename_length, extra_length = fields[-2:]
    data_offset = info.header_offset + 30 + filename_length + extra_length
    compressed = _parallel_range(url, data_offset, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"unsupported ZIP compression {info.compress_type} for {info.filename}"
        )
    if len(payload) != info.file_size:
        raise ValueError(f"uncompressed size mismatch for {info.filename}")
    if binascii.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return payload


def _member_data_offset(url: str, info: zipfile.ZipInfo) -> int:
    header = _request_range(url, info.header_offset, info.header_offset + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"invalid local ZIP header for {info.filename}")
    filename_length, extra_length = fields[-2:]
    return info.header_offset + 30 + filename_length + extra_length


def _uncompressed_chunks(
    url: str, info: zipfile.ZipInfo, compressed_chunk_bytes: int = 8 * 1024 * 1024
):
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f"streaming expects deflate compression: {info.filename}")
    data_offset = _member_data_offset(url, info)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    consumed = 0
    while consumed < info.compress_size:
        length = min(compressed_chunk_bytes, info.compress_size - consumed)
        compressed = _parallel_range(url, data_offset + consumed, length)
        consumed += length
        payload = decompressor.decompress(compressed)
        if payload:
            yield payload
    tail = decompressor.flush()
    if tail:
        yield tail


def _iter_image_records(url: str, info: zipfile.ZipInfo):
    """Incrementally parse only the top-level COCO `images` array."""

    utf8 = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    in_images = False
    for raw_chunk in _uncompressed_chunks(url, info):
        buffer += utf8.decode(raw_chunk)
        if not in_images:
            key_at = buffer.find('"images"')
            if key_at < 0:
                buffer = buffer[-128:]
                continue
            array_at = buffer.find("[", key_at)
            if array_at < 0:
                buffer = buffer[key_at:]
                continue
            in_images = True
            position = array_at + 1

        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                row, end = json_decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if position:
                    buffer = buffer[position:]
                    position = 0
                break
            if not isinstance(row, dict):
                raise ValueError(f"non-object image row in {info.filename}")
            yield row
            position = end
        if len(buffer) > 16 * 1024 * 1024:
            raise RuntimeError(f"streaming JSON buffer grew unexpectedly: {info.filename}")
    raise ValueError(f"images array did not terminate in {info.filename}")


def _load_or_materialize_image_projection(
    url: str, info: zipfile.ZipInfo, destination: Path
) -> list[dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in _iter_image_records(url, info):
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(destination)
    rows: list[dict[str, Any]] = []
    with destination.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _stable_key(seed: int, category: str, row: dict[str, Any]) -> str:
    identity = "|".join(
        [category, str(row["doc_name"]), str(row["file_name"]), str(row["page_no"])]
    )
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()


def _select_candidates(
    rows: list[dict[str, Any]], category: str, count: int, seed: int
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("doc_category") == category and int(row.get("precedence", 0)) == 0
    ]
    ordered = sorted(eligible, key=lambda row: (_stable_key(seed, category, row), row["file_name"]))
    selected: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for row in ordered:
        document = str(row["doc_name"])
        if document in seen_documents:
            continue
        seen_documents.add(document)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{category}: expected {count} distinct documents, got {len(selected)}")
    return selected


def acquire(config_path: Path, project_root: Path, storage_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = config["datasets"]["doclaynet"]
    url = str(spec["core_url"])
    expected_bytes = int(spec["core_expected_bytes"])
    count = int(spec["acquisition_candidate_count_per_stratum"])
    seed = int(config["experiment"]["seed"])
    dataset_dir = storage_root / config["storage"]["relative_root"] / "doclaynet"

    reader = _HTTPRangeReader(url, expected_bytes)
    with zipfile.ZipFile(reader) as archive:
        infos = {info.filename: info for info in archive.infolist()}
    coco_names = sorted(
        name for name in infos if name.startswith("COCO/") and name.endswith(".json")
    )
    if len(coco_names) != 3:
        raise ValueError(f"expected three COCO JSON members, found {coco_names}")

    metadata_records: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for name in coco_names:
        split = Path(name).stem
        destination = dataset_dir / "metadata" / f"{split}_images.jsonl"
        split_rows = _load_or_materialize_image_projection(url, infos[name], destination)
        for row in split_rows:
            row = dict(row)
            row["upstream_split"] = split
            image_rows.append(row)
        metadata_records.append(
            {
                "member": name,
                "projection": "top_level_images_array",
                "source_member_bytes": infos[name].file_size,
                "source_member_crc32": f"{infos[name].CRC:08x}",
                "projected_rows": len(split_rows),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "path": destination.as_posix(),
            }
        )
        print(f"metadata {name}: {len(split_rows)} image rows", flush=True)

    selected: list[dict[str, Any]] = []
    for category in spec["strata"]:
        selected.extend(_select_candidates(image_rows, str(category), count, seed))

    def fetch(row: dict[str, Any]) -> dict[str, Any]:
        member = f"PNG/{row['file_name']}"
        payload = _member_bytes(url, infos[member])
        destination = dataset_dir / "candidates" / str(row["doc_category"]) / str(row["file_name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(payload)
        return {
            "doc_category": row["doc_category"],
            "collection": row["collection"],
            "doc_name": row["doc_name"],
            "page_no": row["page_no"],
            "precedence": row.get("precedence", 0),
            "upstream_split": row["upstream_split"],
            "member": member,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "path": destination.as_posix(),
            "role": "acquisition_candidate_not_final",
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(fetch, selected))
    records.sort(key=lambda row: (row["doc_category"], row["doc_name"], row["page_no"]))
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_path.relative_to(project_root).as_posix(),
        "config_sha256": _sha256(config_path),
        "official_archive_url": url,
        "official_archive_bytes": expected_bytes,
        "full_archive_retained": False,
        "selection_seed": seed,
        "selection_uses_model_output": False,
        "status": "candidate_pool_not_final_source_freeze",
        "metadata": metadata_records,
        "candidate_count": len(records),
        "candidates": records,
    }
    output_path = project_root / config["paths"]["doclaynet_candidate_manifest"]
    _write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire a deterministic DocLayNet candidate pool by HTTP range."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--storage-root", required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    result = acquire(config_path, project_root, Path(args.storage_root))
    print(json.dumps({"candidate_count": result["candidate_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
