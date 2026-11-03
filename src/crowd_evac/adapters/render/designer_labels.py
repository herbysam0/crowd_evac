"""Display labels for the designer panel."""
from __future__ import annotations

FIELD_LABELS: dict[str, str] = {
    "length": "Length (m)",
    "thickness": "Thickness (m)",
    "x": "X (m)",
    "y": "Y (m)",
    "width": "Width (m)",
    "height": "Height (m)",
    "width_m": "Width (m)",
    "capacity_per_second": "Capacity (/s)",
}

ANCHORS: dict[str, str] = {
    "wall": "start point",
    "obstacle": "bottom-left corner",
    "doorway": "centre",
}

ROOM_LABELS: dict[str, str] = {
    "name": "Name",
    "width_m": "Room width (m)",
    "height_m": "Room height (m)",
}
