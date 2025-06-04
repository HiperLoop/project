import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class Body:
    """Class representing a celestial body with mass, radius, position, and velocity."""

    dimension = 2  # Number of dimensions (3D space)

    def __init__(self, name, colour, mass, radius, position, velocity):
        self.name = name
        self.display_colour = colour
        self.mass = mass
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass}, radius={self.radius}, position={self.position}, velocity={self.velocity})"

class Simulation:
    """Class for simulating gravitational interactions between n bodies."""

    def __init__(self, bodies, dimension=2, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.01):
        """Initialize the simulation with a list of bodies, their dimensions, and gravitational constant."""
        self.time_step = time_step  # Time step for the simulation
        self.bodies = bodies
        self.dimension = dimension
        Body.dimension = dimension  # Set the dimension for the Body class
        self.G = G  # Gravitational constant
        self.norming_distance = norming_distance  # Normalization distance in AU
        self.norming_velocity = norming_velocity
        self.unit_norming = norm  # Whether to normalize units
        self.initial_norming()  # Normalize masses and calculate center of mass
        if reverse: self.reverse_velocities()

    def relative_position(self, body1, body2):
        """Calculate the relative position vector from body1 to body2."""
        return body2.position - body1.position

    def twoBody_acceleration(self, body1, body2):
        """Calculate the accelaration fof body one towards body 2."""
        g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        r_vector = self.relative_position(body1, body2)
        distance = np.linalg.norm(r_vector)
        
        if distance == 0:
            return np.zeros(body1.dimension)  # No force if bodies are at the same position
        
        accelration_magnitude = self.G * body2.mass / distance**2
        accelration_vector = (accelration_magnitude / distance) * r_vector
        return accelration_vector

    def accelerations(self):
        """Calculate the accelerations of all bodies in the system due to gravitational interactions."""
        accs = np.zeros((len(self.bodies), self.dimension))
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i < j:
                    accs[i] += self.twoBody_acceleration(body1, body2)
                    accs[j] += self.twoBody_acceleration(body2, body1)
        return accs

    def normMasses(self):
        """Normalize the masses of the bodies to a common scale."""
        total_mass = sum(body.mass for body in self.bodies)
        if total_mass == 0:
            raise ValueError("Total mass of bodies cannot be zero for normalization.")
        for body in self.bodies:
            body.mass /= total_mass

    def centre_of_mass(self):
        """Calculate the center of mass of the system. DO nroming of masses beforehand!"""
        COM_position = np.zeros(self.dimension)
        for body in self.bodies:
            COM_position += body.mass * body.position
        #COM_position = np.add([body.mass * body.position] for body in bodies)
        print("Center of mass position:", repr(COM_position))
        return COM_position

    def relative_positions(self, COM_position):
        """calculate the realtive positions of all bodies with respect to the COM."""
        for body in self.bodies:
            body.position -= COM_position

    def COM_velocity(self):
        """Calculate the velocity of the center of mass of the system. DO mass norming beforehand!"""
        COM_vel = sum(body.mass * body.velocity for body in self.bodies)
        return COM_vel

    def relative_velocities(self, COM_velocity):
        """calculate the realtive velocities of all bodies with respect to the COM."""
        for body in self.bodies:
            body.velocity -= COM_velocity

    def norm_units(self):
        """Normalize the units of position and velocity for the bodies."""
        AU = self.norming_distance
        vel_Earth = self.norming_velocity
        for body in self.bodies:
            body.position /= AU
            body.velocity /= vel_Earth

    def initial_norming(self):
        """Normalize the masses, calculate the center of mass position and velocity, and adjust bodies accordingly."""
        if self.unit_norming: self.norm_units()
        self.normMasses()
        self.COM_position = self.centre_of_mass()
        self.relative_positions(self.COM_position)
        self.COM_vel = self.COM_velocity()
        self.relative_velocities(self.COM_vel)

    def solve_velocities(self, start, sim_duration=10000):
        """Solve the equations of motion for the bodies using the Runge-Kutta method."""
        def equations_of_motion(t, y):
            d = self.dimension
            positions = y[:len(self.bodies) * d].reshape((len(self.bodies), d))
            velocities = y[len(self.bodies) * d:].reshape((len(self.bodies), d))
            
            # Update positions
            for i, body in enumerate(self.bodies):
                body.position = positions[i]
                body.velocity = velocities[i]
            
            # Calculate accelerations
            accs = self.accelerations()
            
            # Return derivatives
            dydt = np.concatenate((velocities.flatten(), accs.flatten()))
            return dydt
        
        initial_conditions = np.concatenate([body.position for body in self.bodies] + [body.velocity for body in self.bodies])
        t_span = (start, self.time_step)
        result = solve_ivp(equations_of_motion, t_span, initial_conditions, vectorized=True)
        return result

    def reverse_velocities(self):
        """Reverse the velocities of the bodies."""
        for body in self.bodies:
            body.velocity *= -1

class Animation:
    """Class for animating the simulation of n bodies."""

    def __init__(self, simulation, plot_size=6, plot_dimensions=1, frame_rate=100, sim_duration=1000):
        """Initialize the animation with the bodies and simulation duration."""
        self.simulation = simulation
        self.plot_size = plot_size
        self.plot_dimensions = plot_dimensions
        self.frame_rate = frame_rate
        self.sim_duration = sim_duration
        self.fig, self.ax = plt.subplots()
        self.fig.set_figheight(plot_size)
        self.fig.set_figwidth(plot_size)
        self.ax.set(xlim=[-plot_dimensions, plot_dimensions], ylim=[-plot_dimensions, plot_dimensions], xlabel='X', ylabel='Y')
        self.global_scats, self.global_lines = self.plot_init()
        self.animate()

    def plot_init(self):
        """Initializes the plot with scatter points and lines for each body."""
        scats = []
        lines = []
        for body in self.simulation.bodies:
            scats.append(self.ax.scatter(0, 0, c=body.display_colour, s=5, label=body.name))
            lines.append(self.ax.plot(body.position[0], body.position[1], c=body.display_colour, alpha=0.2, label=f'{body.name} orbit')[0])
        self.ax.legend()
        return scats, lines

    def update_body_plot(self, body, scat, line):
        """Updates the plot with the current positions of the bodies."""
        data = np.stack([body.position[0], body.position[1]]).T
        scat.set_offsets(data)
        lineData = line.get_data(True)
        line.set_xdata(np.append(lineData[0], body.position[0]))
        line.set_ydata(np.append(lineData[1], body.position[1]))
        return (scat, line)

    def draw_bodies(self, scats, lines):
        """Updates the scatter and line plots for all bodies."""
        for i, body in enumerate(self.simulation.bodies):
            scats[i], lines[i] = self.update_body_plot(body, scats[i], lines[i])
        return scats, lines

    def update(self, frame):
        """Update function for the animation."""
        continuous_evolve = self.simulation.solve_velocities(0, 1000).y
        graphs = [self.draw_bodies(self.global_scats, self.global_lines)]
        return graphs

    def animate(self):
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=1000, interval=1000//self.frame_rate, repeat=False)
        plt.show()

def load_body_from_csv(filename):
        """Load body data from a CSV file."""
        text_bodies = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)[1:]
        bodies = []
        for row in text_bodies:
            name = row[0]
            mass = row[1]
            distance = row[8]
            radius = float(row[2])/2
            velocity = row[12]
            colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            body = Body(name, colour, float(mass), float(radius), [float(distance), float(0)], [float(0), float(velocity)])
            bodies.append(body)
        return bodies

def load_body_from_custom_csv(filename):
    """Load body data from a CSV file."""
    text_bodies = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)[1:]
    bodies = []
    for row in text_bodies:
        name = row[0]
        mass = row[1]
        radius = float(row[2])/2
        colour = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        position = np.array([float(col) for col in row[3:5]])
        velocity = np.array([float(col) for col in row[6:8]])
        print(position, velocity)
        body = Body(name, colour, float(mass), float(radius), position, velocity)
        bodies.append(body)
    return bodies

def figure_eight_configureation():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('custom_objects.csv')[-3:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=3, norm=False, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=3, sim_duration=1000, frame_rate=100)

def solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('custom_objects.csv')[0]
    bodies = [sun] + planets
    print("Loaded bodies:", bodies)
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=40, sim_duration=1000, frame_rate=100)

def main():
    """Main function to run the simulation."""
    # Uncomment the desired simulation configuration
    #figure_eight_configureation()
    solar_system()

main()