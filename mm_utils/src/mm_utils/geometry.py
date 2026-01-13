import numpy as np


class Box3D:
    """A 3D box with position, rotation, and half-extents."""

    def __init__(self, half_extents, position=None, rotation=None):
        self.half_extents = np.array(half_extents)
        assert (self.half_extents > 0).all(), "Half extents must be positive."

        # Store the 8 vertices of the box in local coordinates
        x, y, z = self.half_extents
        # fmt: off
        self.local_vertices = np.array([
            [ x,  y,  z],
            [ x,  y, -z],
            [ x, -y,  z],
            [ x, -y, -z],
            [-x,  y,  z],
            [-x,  y, -z],
            [-x, -y,  z],
            [-x, -y, -z]])
        # fmt: on

        self.update_pose(position, rotation)

    def update_pose(self, position=None, rotation=None):
        """Update the box's position and rotation.

        Args:
            position (ndarray, optional): 3D position vector. Defaults to [0, 0, 0].
            rotation (ndarray, optional): 3x3 rotation matrix. Defaults to identity matrix.
        """
        if position is None:
            position = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)

        self.position = np.array(position)
        self.rotation = np.array(rotation)

        # Transform vertices to world coordinates
        self.vertices = self.position + (self.rotation @ self.local_vertices.T).T

    def height(self):
        """Get the height of the box (along z-axis)."""
        return 2 * self.half_extents[2]

    @property
    def width(self):
        """Get the width of the box (along x-axis)."""
        return 2 * self.half_extents[0]

    @property
    def depth(self):
        """Get the depth of the box (along y-axis)."""
        return 2 * self.half_extents[1]
