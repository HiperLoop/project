import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def relative_position(body1, body2):
    """Calculate the relative position vector from body1 to body2."""
    return body2.position - body1.position

def twoBody_acceleration(body1, body2):
    """Calculate the accelaration fof body one towards body 2."""
    g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    G = 1  # Modified gravitational constant
    r_vector = relative_position(body1, body2)
    distance = np.linalg.norm(r_vector)
    
    if distance == 0:
        return np.zeros(body1.dimension)  # No force if bodies are at the same position
    
    accelration_magnitude = G * body2.mass / distance**2
    accelration_vector = (accelration_magnitude / distance) * r_vector
    return accelration_vector

def accelerations(bodies):
    """Calculate the accelerations of all bodies in the system due to gravitational interactions."""
    accs = np.zeros((len(bodies), bodies[0].dimension))
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                accs[i] += twoBody_acceleration(body1, body2)
                accs[j] += twoBody_acceleration(body2, body1)
    return accs

def normMasses(bodies):
    """Normalize the masses of the bodies to a common scale."""
    total_mass = sum(body.mass for body in bodies)
    if total_mass == 0:
        raise ValueError("Total mass of bodies cannot be zero for normalization.")
    for body in bodies:
        body.mass /= total_mass

def centre_of_mass(bodies):
    """Calculate the center of mass of the system. DO nroming of masses beforehand!"""
    COM_position = np.zeros(bodies[0].dimension)
    for body in bodies:
        COM_position += body.mass * body.position
    #COM_position = np.add([body.mass * body.position] for body in bodies)
    print("Center of mass position:", repr(COM_position))
    return COM_position

def relative_positions(bodies, COM_position):
    """calculate the realtive positions of all bodies with respect to the COM."""
    for body in bodies:
        body.position -= COM_position

def COM_velocity(bodies):
    """Calculate the velocity of the center of mass of the system. DO mass norming beforehand!"""
    COM_vel = sum(body.mass * body.velocity for body in bodies)
    return COM_vel

def relative_velocities(bodies, COM_velocity):
    """calculate the realtive velocities of all bodies with respect to the COM."""
    for body in bodies:
        body.velocity -= COM_velocity

def norm_units(bodies):
    """Normalize the units of position and velocity for the bodies."""
    AU = 1.496e11  # Astronomical Unit in meters
    vel_Earth = 29780  # Earth's orbital velocity in m/s
    for body in bodies:
        body.position /= AU
        body.velocity /= vel_Earth

def initial_norming(bodies):
    """Normalize the masses, calculate the center of mass position and velocity, and adjust bodies accordingly."""
    norm_units(bodies)
    normMasses(bodies)
    COM_position = centre_of_mass(bodies)
    relative_positions(bodies, COM_position)
    COM_vel = COM_velocity(bodies)
    relative_velocities(bodies, COM_vel)

def solve_velocities(bodies, start, dt, sim_duration=10000):
    """Solve the equations of motion for the bodies using the Runge-Kutta method."""
    def equations_of_motion(t, y):
        d = bodies[0].dimension
        positions = y[:len(bodies) * d].reshape((len(bodies), d))
        velocities = y[len(bodies) * d:].reshape((len(bodies), d))
        
        # Update positions
        for i, body in enumerate(bodies):
            body.position = positions[i]
            body.velocity = velocities[i]
        
        # Calculate accelerations
        accs = accelerations(bodies)
        
        # Return derivatives
        dydt = np.concatenate((velocities.flatten(), accs.flatten()))
        return dydt
    
    initial_conditions = np.concatenate([body.position for body in bodies] + [body.velocity for body in bodies])
    t_span = (start, dt)
    result = solve_ivp(equations_of_motion, t_span, initial_conditions, vectorized=True)
    return result

#mass of earth in variable x
x = 5.972e24  # kg
#mass of moon in variable y
y = 7.348e22  # kg
test_earth = Body("Earth", 'green', x, 1, [1.0, 0], [0.0, 1])
test_earth2 = Body("Earth2", 'green', 5.972e24, 6371e3, [100000], [0])
test_moon = Body("Moon", 'blue', y, 1, [1.002569, 0], [0, 1.033])
test_moon2 = Body("Moon", 'blue', y, 1, [-1.0, 0], [0, 1.033])
test_sun = Body("Sun", 'red', 1.989e30, 1, [0.0, 0], [0.0, 0])
test_mars = Body("Mars", 'orange', 6.4171e23, 1, [1.524, 0], [0, 0.81])

bodies = [test_earth, test_sun, test_mars]
initial_norming(bodies)

fig, ax = plt.subplots()

# drawing
dim = 3
ax.set(xlim=[-dim, dim], ylim=[-dim, dim], xlabel='X', ylabel='Y')
fig.set_figheight(6)
fig.set_figwidth(6)
ax.legend()

def plot_init(bodies):
    """Initializes the plot with scatter points and lines for each body."""
    scats = []
    lines = []
    for body in bodies:
        scats.append(ax.scatter(0, 0, c=body.display_colour, s=5, label=body.name))
        lines.append(ax.plot(body.position[0], body.position[1], c=body.display_colour, alpha=0.2, label=f'{body.name} orbit')[0])
    return scats, lines

#creates the scatter and line objects for each body
global_scats, global_lines = plot_init(bodies)

def update_body_plot(body, scat, line):
    """Updates the plot with the current positions of the bodies."""
    data = np.stack([body.position[0], body.position[1]]).T
    scat.set_offsets(data)
    lineData = line.get_data(True)
    line.set_xdata(np.append(lineData[0], body.position[0]))
    line.set_ydata(np.append(lineData[1], body.position[1]))
    return (scat, line)

def draw_bodies(bodies, scats, lines):
    """Updates the scatter and line plots for all bodies."""
    for i, body in enumerate(bodies):
        scats[i], lines[i] = update_body_plot(body, scats[i], lines[i])
    return scats, lines

def update(frame):
    """Update function for the animation."""
    continuous_evolve = solve_velocities(bodies, 0, 0.01, sim_duration=1000).y
    graphs = [draw_bodies(bodies, global_scats, global_lines)]
    return graphs


ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=10)
plt.show()