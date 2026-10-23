"""Bounded, in-memory ZIP ingestion. Uploaded code is never executed."""

import io
from pathlib import PurePosixPath
import stat
import zipfile


MAX_UPLOAD = 10 * 1024 * 1024
MAX_EXPANDED = 30 * 1024 * 1024
MAX_FILES = 200
MAX_FILE_BYTES = 200_000
SKIPPED_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", "__MACOSX"}


def read_repository_zip(data: bytes) -> tuple[dict[str, str], list[str]]:
    if len(data) > MAX_UPLOAD:
        raise ValueError("ZIP exceeds the 10 MB upload limit.")
    sources: dict[str, str] = {}
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 3000 or sum(entry.file_size for entry in entries) > MAX_EXPANDED:
                raise ValueError("Archive expands beyond the allowed size or entry count.")
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename:
                    raise ValueError("Archive contains an unsafe path.")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Symbolic links are not accepted.")
                if entry.flag_bits & 1:
                    raise ValueError("Encrypted archives are not supported.")
                if entry.is_dir() or path.suffix != ".py":
                    continue
                if any(part in SKIPPED_PARTS or part.startswith(".") for part in path.parts):
                    continue
                if entry.file_size > MAX_FILE_BYTES:
                    warnings.append(f"Skipped large file: {path}")
                    continue
                if len(sources) >= MAX_FILES:
                    raise ValueError("Repository exceeds the 200 Python-file limit.")
                try:
                    sources[str(path)] = archive.read(entry).decode("utf-8-sig")
                except UnicodeError:
                    warnings.append(f"Skipped non-UTF-8 file: {path}")
    except zipfile.BadZipFile as error:
        raise ValueError("The upload is not a readable ZIP archive.") from error
    if not sources:
        raise ValueError("No readable Python files were found.")
    paths = [PurePosixPath(path) for path in sources]
    if all(len(path.parts) > 1 for path in paths) and len({path.parts[0] for path in paths}) == 1:
        sources = {"/".join(PurePosixPath(path).parts[1:]): source for path, source in sources.items()}
    return sources, warnings
