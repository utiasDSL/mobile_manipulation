"""
Cost Function Registry for MPC.

This registry provides a centralized way to register and create cost functions,
replacing hardcoded cost selection logic with a more flexible, maintainable approach.
"""

from typing import Dict, Type

from mm_control.MPCCostFunctions import (
    BasePos2CostFunction,
    BasePos3CostFunction,
    BasePoseSE2CostFunction,
    BaseVel2CostFunction,
    BaseVel3CostFunction,
    ControlEffortCostFunction,
    CostFunctions,
    EEPos3BaseFrameCostFunction,
    EEPos3CostFunction,
    EEPoseSE3BaseFrameCostFunction,
    EEPoseSE3CostFunction,
    EEVel3CostFunction,
    RegularizationCostFunction,
)


class CostFunctionRegistry:
    """Registry for cost function classes.

    This registry maps cost function names to their classes, allowing
    dynamic creation of cost functions from configuration.
    """

    _registry: Dict[str, Type[CostFunctions]] = {}

    @classmethod
    def register(cls, name: str, cost_class: Type[CostFunctions]) -> None:
        """Register a cost function class.

        Args:
            name: Name identifier for the cost function (e.g., "EEPos3")
            cost_class: The cost function class to register
        """
        cls._registry[name] = cost_class

    @classmethod
    def create(cls, name: str, robot_model, params: dict):
        """Create a cost function instance.

        Args:
            name: Name of the cost function to create
            robot_model: Robot model instance
            params: Parameters for the cost function

        Returns:
            Instance of the requested cost function

        Raises:
            ValueError: If the cost function name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown cost function: '{name}'. " f"Available: {available}"
            )
        return cls._registry[name](robot_model, params)

    @classmethod
    def list_available(cls) -> list:
        """List all registered cost function names.

        Returns:
            List of registered cost function names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a cost function is registered.

        Args:
            name: Cost function name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry


# Auto-register all existing cost functions
CostFunctionRegistry.register("EEPos3", EEPos3CostFunction)
CostFunctionRegistry.register("EEPos3BaseFrame", EEPos3BaseFrameCostFunction)
CostFunctionRegistry.register("EEPoseSE3", EEPoseSE3CostFunction)
CostFunctionRegistry.register("EEPoseSE3BaseFrame", EEPoseSE3BaseFrameCostFunction)
CostFunctionRegistry.register("EEVel3", EEVel3CostFunction)
CostFunctionRegistry.register("BasePos2", BasePos2CostFunction)
CostFunctionRegistry.register("BasePos3", BasePos3CostFunction)
CostFunctionRegistry.register("BasePoseSE2", BasePoseSE2CostFunction)
CostFunctionRegistry.register("BaseVel2", BaseVel2CostFunction)
CostFunctionRegistry.register("BaseVel3", BaseVel3CostFunction)
CostFunctionRegistry.register("ControlEffort", ControlEffortCostFunction)
CostFunctionRegistry.register("Regularization", RegularizationCostFunction)
