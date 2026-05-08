"""RFI mask construction + per-data-dict mask plumbing.

Project rule: there is no built-in default mask. Callers always supply
their own RFI ranges (or load the versioned static config). The
previous ``DEFAULT_RFI_MASK`` constant from ``casm-bf-imaging`` is
intentionally not reproduced.

Usage
-----

>>> mask = RFIMask(bad_ranges_mhz=[(395.0, 405.0), (450.0, 455.0)])
>>> good = mask(freqs_mhz)         # bool array, True = good
>>> mask.flag_bins(freqs_mhz)      # bool array, True = bad

>>> static = RFIMask.from_static()         # latest versioned config
>>> static = RFIMask.from_static(version=1)

>>> apply_rfi_mask(data, static)           # populates data['freq_mask*']
>>> apply_rfi_mask(data, static, dynamic_mask)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class RFIMask:
    """Boolean frequency mask built from contaminated MHz ranges.

    Parameters
    ----------
    bad_ranges_mhz : list of (lo, hi) tuples in MHz
        Inclusive, in MHz. Required (no default).
    label : str, optional
        Human-readable name (e.g. ``"static_v1"``, ``"dynamic_2026-05-07"``).
    """

    bad_ranges_mhz: list = field(default_factory=list)
    label: str = ""

    def __post_init__(self):
        if self.bad_ranges_mhz is None:
            self.bad_ranges_mhz = []
        cleaned = []
        for entry in self.bad_ranges_mhz:
            if isinstance(entry, dict):
                lo, hi = entry["lo"], entry["hi"]
            else:
                lo, hi = entry
            if hi < lo:
                raise ValueError(f"Bad range ({lo}, {hi}) MHz: hi < lo.")
            cleaned.append((float(lo), float(hi)))
        self.bad_ranges_mhz = cleaned

    def flag_bins(self, freqs_mhz) -> np.ndarray:
        """True where the channel is contaminated (bad)."""
        freqs = np.asarray(freqs_mhz, dtype=float)
        bad = np.zeros(freqs.shape, dtype=bool)
        for lo, hi in self.bad_ranges_mhz:
            bad |= (freqs >= lo) & (freqs <= hi)
        return bad

    def __call__(self, freqs_mhz) -> np.ndarray:
        """True where the channel is good (NOT contaminated)."""
        return ~self.flag_bins(freqs_mhz)

    # --------------------------------------------------------------- io

    @classmethod
    def from_json(cls, path) -> "RFIMask":
        """Load a static-RFI JSON config (versioned in repo)."""
        path = Path(path)
        with open(path) as f:
            payload = json.load(f)
        bands = payload.get("bands_mhz", [])
        return cls(
            bad_ranges_mhz=bands,
            label=f"{payload.get('site', 'unknown')}_static_v{payload.get('version', '?')}",
        )

    @classmethod
    def from_static(cls, version: int | None = None) -> "RFIMask":
        """Load the static RFI config shipped with this repo.

        Parameters
        ----------
        version : int, optional
            Specific version to load. ``None`` (default) picks the
            highest version on disk under ``casm_vis_analysis/configs``.
        """
        candidates = sorted(_CONFIG_DIR.glob("rfi_static_v*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No rfi_static_v*.json found in {_CONFIG_DIR}"
            )
        if version is None:
            path = candidates[-1]
        else:
            wanted = _CONFIG_DIR / f"rfi_static_v{version}.json"
            if not wanted.exists():
                raise FileNotFoundError(
                    f"version={version} not found; available: "
                    f"{[p.stem for p in candidates]}"
                )
            path = wanted
        return cls.from_json(path)


# ---------------------------------------------------------------------------
# Per-data-dict mask plumbing
# ---------------------------------------------------------------------------


def _resolve_to_bool(mask, freqs_mhz):
    """Coerce an RFIMask / bool array / None into a bool flag array (True=bad)."""
    if mask is None:
        return None
    if isinstance(mask, RFIMask):
        return mask.flag_bins(freqs_mhz)
    arr = np.asarray(mask)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    if arr.shape != freqs_mhz.shape:
        raise ValueError(
            f"mask shape {arr.shape} doesn't match freq axis {freqs_mhz.shape}. "
            f"Pass an RFIMask, a bool array of length {len(freqs_mhz)}, or None."
        )
    return arr


def apply_rfi_mask(data, static=None, *, dynamic=None):
    """Attach RFI flags to a ``data`` dict in place. Does NOT modify ``data['vis']``.

    Populates:
      ``data['freq_mask_static']``  — bool array (F,), True = flagged.
      ``data['freq_mask_dynamic']`` — bool array (F,) or ``None``.
      ``data['freq_mask']``         — OR of populated source masks.

    Forward-compat: when 2D (T, F) masks land later, the same keys hold
    them and consumers route through :func:`_freq_mask_for_channel` to
    pick the right slice. Today only 1D is supported.

    Parameters
    ----------
    data : VisibilityResult or dict-like
        Must expose ``freq_mhz``. Mutated in place.
    static : RFIMask or bool array, optional
        Persistent flagged bands.
    dynamic : RFIMask or bool array, optional
        Per-observation auto-detected flags (e.g. SK or sigma-clip).
    """
    freqs = np.asarray(_dict_or_attr(data, "freq_mhz"), dtype=float)
    static_b = _resolve_to_bool(static, freqs)
    dynamic_b = _resolve_to_bool(dynamic, freqs)

    if static_b is None and dynamic_b is None:
        combined = np.zeros(freqs.shape, dtype=bool)
    elif static_b is None:
        combined = dynamic_b
    elif dynamic_b is None:
        combined = static_b
    else:
        combined = static_b | dynamic_b

    _dict_or_attr_set(data, "freq_mask_static", static_b)
    _dict_or_attr_set(data, "freq_mask_dynamic", dynamic_b)
    _dict_or_attr_set(data, "freq_mask", combined)
    return data


def _freq_mask_for_channel(data, t=None) -> np.ndarray | None:
    """Return per-channel flag array (True = flagged) at time index ``t``.

    Forward-compat for 2D (T, F) masks: today the stored mask is 1D so
    ``t`` is ignored. When 2D arrives we slice ``mask[t]`` here without
    changing call sites.
    """
    m = _dict_or_attr(data, "freq_mask", default=None)
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim == 1:
        return m
    if m.ndim == 2:
        return m[t] if t is not None else m.any(axis=0)
    raise ValueError(f"freq_mask must be 1D or 2D, got {m.shape}")


def _dict_or_attr(obj, key, default="<raise>"):
    """Read either dict[key] or obj.key (handles VisibilityResult)."""
    if hasattr(obj, "__getitem__"):
        try:
            return obj[key]
        except (KeyError, TypeError):
            pass
    if hasattr(obj, key):
        return getattr(obj, key)
    if default == "<raise>":
        raise KeyError(key)
    return default


def _dict_or_attr_set(obj, key, value):
    """Write either dict[key] = value or setattr(obj, key, value)."""
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        obj[key] = value
    except (TypeError, KeyError):
        setattr(obj, key, value)
