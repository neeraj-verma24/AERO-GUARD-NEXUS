"""Validated environment-backed runtime settings for AERO-GUARD NEXUS."""
from __future__ import annotations

import math
import os


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
