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
    print("Max Earth distance:", np.linalg.norm(body2.position - body1.position))
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
    accs = np.zeros((len(bodies), bodies[0].dimension))
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                accs[i] += twoBody_acceleration(body1, body2)
                accs[j] += twoBody_acceleration(body2, body1)
    #print("Forces calculated:", forces)
    return accs

def normMasses(bodies):
    """Normalize the masses of the bodies to a common scale."""
    total_mass = sum(body.mass for body in bodies)
    if total_mass == 0:
        raise ValueError("Total mass of bodies cannot be zero for normalization.")
    for body in bodies:
        body.mass /= total_mass
    print("Normalized masses:", [body.mass for body in bodies])

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

#earth = Body("Earth", 5.972e24, 6371e3, [0, 0, 0], [0, 0, 0])
#earth2 = Body("Earth2", 5.972e24, 6371e3, [100000, 0, 0], [0, 0, 0])
#moon = Body("Moon", 7.348e22, 1737e3, [384400e3, 0, 0], [0, 1022, 0])

bodies = [test_earth, test_sun, test_mars]
initial_norming(bodies)
#bodies = [earth, moon]

""" evolve = solve_velocities(bodies,0,  0.01, sim_duration=1000000)
print("Evolution result:", evolve)
print(evolve.y.shape)
print("Final positions:", [body.position for body in bodies]) """

fig, ax = plt.subplots()

#scat = ax.scatter(0, 0, c="r", s=5, label=f'sun')
#scat2 = ax.scatter(0, 0, c="g", s=5, label=f'earth')
#scat3 = ax.scatter(0, 0, c="b", s=5, label=f'moon')
#line = ax.plot(test_moon.position[0], test_moon.position[1], c="k", alpha=0.2, label=f'earth orbit')[0]
dim = 3
ax.set(xlim=[-dim, dim], ylim=[-dim, dim], xlabel='X', ylabel='Y')
fig.set_figheight(6)
fig.set_figwidth(6)
ax.legend()

def plot_init(bodies):
    scats = []
    lines = []
    for body in bodies:
        scats.append(ax.scatter(0, 0, c=body.display_colour, s=5, label=body.name))
        lines.append(ax.plot(body.position[0], body.position[1], c=body.display_colour, alpha=0.2, label=f'{body.name} orbit')[0])
    return scats, lines

global_scats, global_lines = plot_init(bodies)

def update_body_plot(body, scat, line):
    """Draws body and its orbit in the plot."""
    data = np.stack([body.position[0], body.position[1]]).T
    scat.set_offsets(data)
    lineData = line.get_data(True)
    line.set_xdata(np.append(lineData[0], body.position[0]))
    line.set_ydata(np.append(lineData[1], body.position[1]))
    return (scat, line)

def draw_bodies(bodies, scats, lines):
    for i, body in enumerate(bodies):
        scats[i], lines[i] = update_body_plot(body, scats[i], lines[i])
    return scats, lines

def update(frame):
    #continuous_evolve = solve_COM(bodies).y
    continuous_evolve = solve_velocities(bodies, 0, 0.01, sim_duration=1000).y
    #print(continuous_evolve[2][-1], continuous_evolve[3][-1])
    #all_evolves += (continuous_evolve)
    # for each frame, update the data stored on each artist.
    """ x = continuous_evolve[2][-1]
    y = continuous_evolve[3][-1]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # for each frame, update the data stored on each artist.
    x2 = continuous_evolve[0][-1]
    y2 = continuous_evolve[1][-1]
    # update the scatter plot:
    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)
    # for each frame, update the data stored on each artist.
    x3 = continuous_evolve[4][-1]
    y3 = continuous_evolve[5][-1]
    # update the scatter plot:
    data3 = np.stack([x3, y3]).T
    scat3.set_offsets(data3)
    # for each frame, update the data stored on each artist.
    lineData = line.get_data(True)
    line.set_xdata(np.append(lineData[0], continuous_evolve[0][::100]))
    line.set_ydata(np.append(lineData[1], continuous_evolve[1][::100])) """
    graphs = [draw_bodies(bodies, global_scats, global_lines)]
    return graphs


ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=10)
plt.show()