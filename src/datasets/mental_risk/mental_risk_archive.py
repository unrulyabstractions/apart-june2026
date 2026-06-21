"""Password-protected extraction of the MentalRiskES corpus archives.

The official archives are encrypted (ZipCrypto/AES). The access password is
never stored in the repo: it must come from the environment or a password file
that the researcher obtains after signing the corpus usage agreement.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyzipper

from src.common.file_io import ensure_dir
from src.common.logging import log

PASSWORD_ENV = "MENTALRISK_ZIP_PASSWORD"


def resolve_password(password_file: Path | None = None) -> bytes:
    """Resolve the archive password from env var, then file, else raise.

    Returns the password as bytes (pyzipper's `setpassword` expects bytes).
    """
    env_value = os.environ.get(PASSWORD_ENV)
    if env_value:
        return env_value.encode("utf-8")
    if password_file is not None and Path(password_file).exists():
        return Path(password_file).read_text(encoding="utf-8").strip().encode("utf-8")
    raise RuntimeError(
        "MentalRiskES archives are encrypted. Set the access password in the "
        f"{PASSWORD_ENV} environment variable (or pass a password_file). Obtain "
        "the password by requesting corpus access from the MentalRiskES organizers."
    )


def extract_archive(zip_path: Path, out_dir: Path, password: bytes) -> Path:
    """Extract one encrypted zip into out_dir, recursing into nested zips.

    Nested zips share the same password and are deleted after extraction so the
    output tree contains only the final corpus files.
    """
    zip_path, out_dir = Path(zip_path), ensure_dir(Path(out_dir))
    with pyzipper.AESZipFile(zip_path) as zf:
        zf.setpassword(password)
        zf.extractall(out_dir)
    for nested in out_dir.rglob("*.zip"):
        extract_archive(nested, nested.parent, password)
        nested.unlink()
    return out_dir


def extract_corpus(corpus_dir: Path, out_dir: Path, password: bytes) -> Path:
    """Extract every top-level *.zip under corpus_dir into out_dir."""
    corpus_dir, out_dir = Path(corpus_dir), ensure_dir(Path(out_dir))
    archives = sorted(corpus_dir.glob("*.zip"))
    if not archives:
        log(f"[mental_risk] no *.zip archives found under {corpus_dir}")
    for archive in archives:
        log(f"[mental_risk] extracting {archive.name}")
        extract_archive(archive, out_dir, password)
    return out_dir
