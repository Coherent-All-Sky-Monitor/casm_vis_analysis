"""Pull the CAsMan database snapshot from GitHub Releases.

CAsMan performs an auto-sync check at import time, but that check is inert
from a downstream consumer's point of view — it doesn't surface whether a
newer snapshot exists or force a refresh. This module replaces it with an
explicit, controllable pull: call `pull_casman()` to fetch (or verify) the
latest snapshot, or `pull_casman(offline=True)` to skip the network
entirely and report on whatever is already on disk.

Databases land in `~/.local/share/casman/databases/` (or
`$XDG_DATA_HOME/casman/databases` when `XDG_DATA_HOME` is set) — the same
location CAsMan's `GitHubSyncManager` uses.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _local_db_dir() -> Path:
    """Mirror CAsMan's GitHubSyncManager XDG path resolution.

    Used as a fallback when casman isn't importable (offline installs) or
    when the sync manager can't be constructed.
    """
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base_dir = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base_dir / "casman" / "databases"


def _resolve_local_db_dir() -> Path:
    """Get the local database dir, preferring casman's own resolution."""
    try:
        from casman.database.github_sync import get_github_sync_manager
        mgr = get_github_sync_manager()
        if mgr is not None:
            return mgr.local_db_dir
    except Exception:
        pass
    return _local_db_dir()


def _read_local_metadata(db_dir: Path) -> tuple[str | None, str | None]:
    """Read (release_name, timestamp) from `.sync_metadata.json`, if present."""
    metadata_file = db_dir / ".sync_metadata.json"
    if not metadata_file.exists():
        return None, None
    try:
        with open(metadata_file) as f:
            data = json.load(f)
        return data.get("release_name"), data.get("timestamp")
    except Exception:
        return None, None


def _offline_result() -> dict:
    db_dir = _resolve_local_db_dir()
    release_name, timestamp = _read_local_metadata(db_dir)
    return {"release_name": release_name, "timestamp": timestamp,
            "downloaded": False, "source": "local"}


def pull_casman(*, offline: bool = False, force: bool = False) -> dict:
    """Fetch the latest CAsMan database snapshot from GitHub Releases.

    Parameters
    ----------
    offline : bool
        When True, never touch the network — just report on the local
        `.sync_metadata.json`.
    force : bool
        When True, re-download even if the local copy already matches the
        latest checksum (passed through to `GitHubSyncManager.download_databases`).

    Returns
    -------
    dict
        ``{'release_name': str | None, 'timestamp': str | None,
        'downloaded': bool, 'source': 'github' | 'local'}``
    """
    if offline:
        return _offline_result()

    try:
        from casman.database.github_sync import get_github_sync_manager
        mgr = get_github_sync_manager()
        if mgr is None:
            raise RuntimeError("get_github_sync_manager() returned None "
                               "(casman not configured)")
        latest = mgr.get_latest_release()
        if latest is None:
            raise RuntimeError("no database snapshots found on CAsMan GitHub releases")

        up_to_date = mgr._is_local_up_to_date(latest)
        mgr.download_databases(snapshot=latest, force=force)

        return {"release_name": latest.release_name,
                "timestamp": latest.timestamp.isoformat(),
                "downloaded": not up_to_date,
                "source": "github"}
    except Exception as e:
        print(f"WARNING: could not reach CAsMan GitHub releases ({e}); "
              f"using local database copy.", file=sys.stderr)
        return _offline_result()
