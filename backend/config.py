"""Validated environment-backed runtime settings for AERO-GUARD NEXUS."""
from __future__ import annotations

import math
import os
from pathlib import Path


def get_float_setting(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Read a finite float from the environment and validate its allowed range."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = float(default)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Environment setting {name} must be a number; received {raw!r}."
            ) from exc

    if not math.isfinite(value):
        raise RuntimeError(f"Environment setting {name} must be a finite number.")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"Environment setting {name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"Environment setting {name} must be <= {maximum}.")
    return value

def get_bool_setting(name: str, default: bool) -> bool:
    """Read a strict boolean environment setting; reject unknown spellings."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"Environment setting {name} must be a boolean "
        "(true/false, yes/no, on/off, or 1/0)."
    )

def get_path_setting(
    name: str,
    default: Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    """Resolve a filesystem path from the environment or a safe default.

    Relative environment paths are resolved from ``relative_to`` when supplied;
    otherwise they are resolved from the current working directory.
    """
    raw = os.getenv(name)
    candidate = Path(raw.strip()).expanduser() if raw and raw.strip() else Path(default)
    if not candidate.is_absolute():
        candidate = (relative_to if relative_to is not None else Path.cwd()) / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"Environment setting {name} is not a valid path.") from exc
    if resolved.exists() and resolved.is_dir():
        raise RuntimeError(f"Environment setting {name} must point to a database file, not a directory.")
    return resolved
