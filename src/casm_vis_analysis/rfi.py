"""RFI mask construction.

Project rule: there is no built-in default mask. Callers always supply
their own RFI ranges (or pass ``None`` for "no masking"). The previous
``DEFAULT_RFI_MASK`` constant from ``casm-bf-imaging`` is intentionally
not reproduced here.

Usage
-----

>>> mask = RFIMask(bad_ranges_mhz=[(395.0, 405.0), (450.0, 455.0)])
>>> good = mask(freqs_mhz)         # bool array, True = good
>>> mask.flag_bins(freqs_mhz)      # bool array, True = bad
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RFIMask:
    """Boolean frequency mask built from contaminated MHz ranges.

    Parameters
    ----------
    bad_ranges_mhz : list of (lo, hi) tuples in MHz
        Inclusive, in MHz. Required (no default).
    """

    bad_ranges_mhz: list = field(default_factory=list)

    def __post_init__(self):
        # Allow None as "no RFI flagged" (callers passing rfi_mask=None).
        if self.bad_ranges_mhz is None:
            self.bad_ranges_mhz = []
        cleaned = []
        for lo, hi in self.bad_ranges_mhz:
            if hi < lo:
                raise ValueError(
                    f"Bad range ({lo}, {hi}) MHz: hi < lo."
                )
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
