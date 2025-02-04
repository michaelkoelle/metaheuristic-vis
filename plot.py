import io
from dataclasses import dataclass
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point:
    x: float
    y: float
    size: int = 5
    style: str = "wo"


class OptimizationVisualizer:
    def __init__(self, function, x_bounds, y_bounds, interval=200):
        self.function = function
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.interval = interval  # Interval between frames in milliseconds
        self.last_best_point = None
        self.frames = []
        # Set figure size and DPI
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(x_bounds)
        self.ax.set_ylim(y_bounds)
        self._plot_function()

    def _plot_function(self):
        """Plot the background contour of the function."""
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], 400)
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], 400)
        X, Y = np.meshgrid(x, y)
        Z = self.function({"x": X, "y": Y})

        levels = np.linspace(np.min(Z), np.max(Z), 50)
        contour = self.ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
        self.fig.colorbar(contour, ax=self.ax)

    def capture_frame(self, points: List[Point]):
        """Capture the current frame with the solution points."""
        point_axs = []

        for point in points:
            point_ax, = self.ax.plot(point.x, point.y, point.style, markersize=point.size)
            point_axs.append(point_ax)

        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.imread(buf)
        self.frames.append(frame)

        for point_ax in point_axs:
            point_ax.remove()

    def capture_frame_with_velocities(self, positions: List[Point], velocities: List[Point]):
        """Capture the current frame with particles and their velocities visualized as arrows."""
        point_axs = []
        arrow_ax = None

        # Plot the particles
        for point in positions:
            point_ax, = self.ax.plot(point.x, point.y, point.style, markersize=point.size)
            point_axs.append(point_ax)

        # Prepare the velocities for quiver plot
        X = [p.x for p in positions]
        Y = [p.y for p in positions]
        U = [v.x for v in velocities]
        V = [v.y for v in velocities]

        # Plot the velocities as arrows
        arrow_ax = self.ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=0.4, color='yellow', width=0.003)

        # Capture the frame
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        frame = imageio.imread(buf)
        self.frames.append(frame)

        # Remove the plotted points and arrows
        for point_ax in point_axs:
            point_ax.remove()
        if arrow_ax:
            arrow_ax.remove()

    def create_gif(self, filename='gifs/optimization.gif'):
        """Generate a GIF from the captured frames."""
        imageio.mimsave(filename, self.frames, fps=1000 / self.interval)
        plt.close(self.fig)
        print(f"GIF saved as {filename}")
