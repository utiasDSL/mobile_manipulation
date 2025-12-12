"""
Planner Registry for Task Management.

This registry provides a centralized way to register and create planners,
replacing string-based lookup with explicit registration for better type safety.
"""

from typing import Dict, Type

from mm_plan.BasePlanner import (
    BasePoseTrajectoryLine,
    BasePosTrajectoryCircle,
    BasePosTrajectoryLine,
    BaseSingleWaypoint,
    BaseSingleWaypointPose,
    ROSTrajectoryPlanner,
    ROSTrajectoryPlannerOnDemand,
)
from mm_plan.EEPlanner import (
    EELookAhead,
    EELookAheadWorld,
    EEPos3WaypointOnDemand,
    EEPoseSE3Waypoint,
    EEPoseSE3WaypointOnDemand,
    EEPosTrajectoryCircle,
    EEPosTrajectoryLine,
    EESimplePlanner,
    EESimplePlannerBaseFrame,
)
from mm_plan.PlanBaseClass import Planner


class PlannerRegistry:
    """Registry for planner classes.

    This registry maps planner type names to their classes, allowing
    dynamic creation of planners from configuration with better error handling.
    """

    _registry: Dict[str, Type[Planner]] = {}

    @classmethod
    def register(cls, name: str, planner_class: Type[Planner]) -> None:
        """Register a planner class.

        Args:
            name: Name identifier for the planner (e.g., "BaseSingleWaypoint")
            planner_class: The planner class to register
        """
        cls._registry[name] = planner_class

    @classmethod
    def create(cls, config: dict) -> Planner:
        """Create a planner instance from configuration.

        Args:
            config: Planner configuration dictionary containing "planner_type"

        Returns:
            Instance of the requested planner

        Raises:
            ValueError: If the planner type is not registered or missing
        """
        if "planner_type" not in config:
            raise ValueError("Configuration missing 'planner_type' field")

        planner_type = config["planner_type"]
        if planner_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown planner type: '{planner_type}'. " f"Available: {available}"
            )
        return cls._registry[planner_type](config)

    @classmethod
    def list_available(cls) -> list:
        """List all registered planner type names.

        Returns:
            List of registered planner type names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a planner type is registered.

        Args:
            name: Planner type name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry


# Auto-register all existing planners
# Base planners
PlannerRegistry.register("BaseSingleWaypoint", BaseSingleWaypoint)
PlannerRegistry.register("BaseSingleWaypointPose", BaseSingleWaypointPose)
PlannerRegistry.register("BasePosTrajectoryCircle", BasePosTrajectoryCircle)
PlannerRegistry.register("BasePosTrajectoryLine", BasePosTrajectoryLine)
PlannerRegistry.register("BasePoseTrajectoryLine", BasePoseTrajectoryLine)
PlannerRegistry.register("ROSTrajectoryPlanner", ROSTrajectoryPlanner)
PlannerRegistry.register("ROSTrajectoryPlannerOnDemand", ROSTrajectoryPlannerOnDemand)

# End-effector planners
PlannerRegistry.register("EESimplePlanner", EESimplePlanner)
PlannerRegistry.register("EESimplePlannerBaseFrame", EESimplePlannerBaseFrame)
PlannerRegistry.register("EEPoseSE3Waypoint", EEPoseSE3Waypoint)
PlannerRegistry.register("EEPoseSE3WaypointOnDemand", EEPoseSE3WaypointOnDemand)
PlannerRegistry.register("EEPos3WaypointOnDemand", EEPos3WaypointOnDemand)
PlannerRegistry.register("EEPosTrajectoryCircle", EEPosTrajectoryCircle)
PlannerRegistry.register("EEPosTrajectoryLine", EEPosTrajectoryLine)
PlannerRegistry.register("EELookAhead", EELookAhead)
PlannerRegistry.register("EELookAheadWorld", EELookAheadWorld)
