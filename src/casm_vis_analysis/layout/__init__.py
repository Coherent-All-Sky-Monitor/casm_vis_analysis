"""Antenna layout pipeline: positions + wiring -> consumer CSV.

Two stages, both available as Python API and CLI:

* `run_sync_wiring`  /  `casm-sync-wiring`   — regenerate `casm_wiring.csv`
   from CAsMan plus a manual-overrides file.
* `run_build_layout` /  `casm-build-layout`  — join wiring with positions,
   compute ENU, write the AntennaMapping-compatible consumer CSV.
"""

from casm_vis_analysis.layout.build import run_build_layout
from casm_vis_analysis.layout.sync import run_sync_wiring

__all__ = ["run_build_layout", "run_sync_wiring"]
