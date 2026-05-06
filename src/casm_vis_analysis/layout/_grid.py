"""Helpers for CAsMan grid_code parsing."""

import re

# CAsMan grid_code format: <array_id><N|C|S><nnn>E<nn>
# Examples: CN021E03 -> plank N21, element E3
#           CC000E01 -> plank C00, element E1
GRID_RE = re.compile(r"^[A-Z]([NCS])(\d{3})E(\d{2})$")


def parse_grid_code(grid_code):
    """Decode a CAsMan grid_code into ('<plank>', '<element>') or None."""
    if not grid_code:
        return None
    m = GRID_RE.match(grid_code)
    if not m:
        return None
    letter = m.group(1)
    plank_num = int(m.group(2))
    element_num = int(m.group(3))
    return f"{letter}{plank_num:02d}", f"E{element_num}"
