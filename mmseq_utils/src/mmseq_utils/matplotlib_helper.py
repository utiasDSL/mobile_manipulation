import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmseq_utils.color_profile import *

def plot_square_box(dimension, center, ax=None):
    # Calculate half the dimension for correct positioning
    half_dim = dimension / 2

    # Extract the center coordinates
    center_x, center_y = center

    # Create a figure and axis
    if ax is None:
        fig, ax = plt.subplots()

    # Define the square using Rectangle from patches
    square = patches.Rectangle(
        (center_x - half_dim, center_y - half_dim),  # Bottom-left corner
        dimension,  # Width
        dimension,  # Height
        edgecolor='darkgrey',  # Border color
        facecolor='lightgrey',  # Fill color
        linewidth=2,  # Border thickness
        alpha=1     # Transparency of the fill
    )

    # Add the square to the plot
    ax.add_patch(square)

    return ax
 
def plot_circle(center, radius, ax=None):
    # Create new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots()

    # Define the circle using Circle from patches
    circle = patches.Circle(
        center,       # Center of the circle (x, y)
        radius,       # Radius of the circle
        edgecolor='black',    # Circle border color
        facecolor=YELLOW,     # Fill color
        linewidth=2,          # Border thickness
        alpha=0.75             # Transparency of the fill
    )

    # Add the circle to the axes
    ax.add_patch(circle)

    return ax

def plot_cross(point, length=1.0, linewidth=3, color='red', ax=None):
    """
    Plots a right-angled cross at a given point.

    Parameters:
    - point (tuple): Coordinates of the cross center (x, y).
    - length (float): Length of each arm of the cross.
    - linewidth (int): Thickness of the cross lines.
    - color (str): Color of the cross lines.
    - ax (matplotlib.axes.Axes, optional): Existing axes to draw the cross on.
    """
    # Create new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate the coordinates for the diagonal lines at 45° and -45°
    x, y = point
    half_length = length / 2
    ax.plot([x - half_length, x + half_length], [y - half_length, y + half_length], 
            color=color, linewidth=linewidth)  # Diagonal line from bottom-left to top-right
    ax.plot([x - half_length, x + half_length], [y + half_length, y - half_length], 
            color=color, linewidth=linewidth)  # Diagonal line from top-left to bottom-right


    return ax
