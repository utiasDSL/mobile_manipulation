"""Enums for planner and control system types."""

from enum import Enum


class RefType(Enum):
    """Type of reference: waypoint or path."""

    WAYPOINT = "waypoint"
    PATH = "path"
