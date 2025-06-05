import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from loaders import load_simulation_from_file
from body import Body

class Animation:
    """Class for animating the simulation of n bodies."""

    def __init__(self, plot_size=6, plot_dimensions=1, frame_rate=100, data_from_file=False, **kwargs):
        """Initialize the animation with the bodies and simulation duration."""
        self.data_from_file = data_from_file
        if data_from_file:
            self.dimension = kwargs.get('dimension', 2)
            Body.dimension = self.dimension
            self.bodies, self.data = load_simulation_from_file(kwargs.get('file_name', None), self.dimension)
        else:
            self.simulation = kwargs.get('simulation', None)
            self.bodies = self.simulation.bodies
            self.dimension = self.simulation.dimension

        self.step = 0
        self.plot_size = plot_size
        self.plot_dimensions = plot_dimensions
        self.frame_rate = frame_rate
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d' if self.dimension == 3 else None)
        self.fig.set_figheight(plot_size)
        self.fig.set_figwidth(plot_size)
        if self.dimension == 3:
            self.ax.set(xlim=[-plot_dimensions, plot_dimensions], ylim=[-plot_dimensions, plot_dimensions], zlim=[-plot_dimensions, plot_dimensions], xlabel='X', ylabel='Y', zlabel='Z')
        else:
            self.ax.set(xlim=[-plot_dimensions, plot_dimensions], ylim=[-plot_dimensions, plot_dimensions], xlabel='X', ylabel='Y')
        self.global_scats, self.global_lines = self.plot_init()

        self.animate()

    def plot_init(self):
        """Initializes the plot with scatter points and lines for each body."""
        scats = []
        lines = []
        for body in self.bodies:
            if self.dimension == 3:
                scats.append(self.ax.scatter(body.position[0], body.position[1], body.position[2], c=body.display_colour, s=5, label=body.name))
                lines.append(self.ax.plot(body.position[0], body.position[1], body.position[2], zdir='z', c=body.display_colour, alpha=0.2, label=f'{body.name} orbit')[0])
            else:
                scats.append(self.ax.scatter(body.position[0], body.position[1], c=body.display_colour, s=5, label=body.name))
                lines.append(self.ax.plot(body.position[0], body.position[1], c=body.display_colour, alpha=0.2, label=f'{body.name} orbit')[0])
        self.ax.legend()
        return scats, lines

    def update_body_plot(self, body, scat, line):
        """Updates the plot with the current positions of the bodies."""
        #data = np.stack([body.position[i] for i in range(body.dimension)]).T
        data = np.array(body.position).reshape(1, 3) if body.dimension == 3 else np.array(body.position).reshape(1, 2)
        if body.dimension == 3:
            scat._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
            scat.set_offsets(data[:, :2])
        else:
            scat.set_offsets(data)
        lineData = line.get_data_3d() if body.dimension == 3 else line.get_data()
        line.set_xdata(np.append(lineData[0], body.position[0]))
        line.set_ydata(np.append(lineData[1], body.position[1]))
        if body.dimension == 3:
            line.set_3d_properties(np.append(lineData[2], body.position[2]))
        return (scat, line)

    def draw_bodies(self, scats, lines):
        """Updates the scatter and line plots for all bodies."""
        for i, body in enumerate(self.bodies):
            scats[i], lines[i] = self.update_body_plot(body, scats[i], lines[i])
        return scats, lines

    def update_bodies(self):
        if self.data_from_file:
            self.step = min(self.step, len(self.data) - 1)
            row = self.data[self.step]
            for i, body in enumerate(self.bodies):
                body.position = [float(row[(i*2) * 3 + j]) for j in range(self.dimension)]
                body.velocity = [float(row[((i*2)+1) * 3 + j]) for j in range(self.dimension)]
        else:
            continuous_evolve = self.simulation.solve_velocities(0).y

    def update(self, frame):
        """Update function for the animation."""
        self.update_bodies()
        graphs = [self.draw_bodies(self.global_scats, self.global_lines)]
        self.step += 1
        return graphs

    def animate(self):
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=100000, interval=1000//self.frame_rate, repeat=True)
        plt.show()