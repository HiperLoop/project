import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Body:
    """Class representing a celestial body with mass, radius, position, and velocity."""

    dimension = 2  # Number of dimensions (3D space)

    def __init__(self, name, mass, radius, position, velocity):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def __repr__(self):
        return f"Body(name={self.name}, mass={self.mass}, radius={self.radius}, position={self.position}, velocity={self.velocity})"
    
def relative_position(body1, body2):
    """Calculate the relative position vector from body1 to body2."""
    return body2.position - body1.position

def relative_force(body1, body2):
    """Calculate the gravitational force exerted by body2 on body1."""
    g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    G = 1  # Modified gravitational constant
    r_vector = relative_position(body1, body2)
    distance = np.linalg.norm(r_vector)
    
    if distance == 0:
        return np.zeros(body1.dimension)  # No force if bodies are at the same position
    
    force_magnitude = G * body1.mass * body2.mass / distance**2
    force_vector = (force_magnitude / distance) * r_vector
    return force_vector

def forces(bodies):
    forces = np.zeros((len(bodies), bodies[0].dimension))
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                forces[i] += relative_force(body1, body2)
                forces[j] -= relative_force(body1, body2)
    #print("Forces calculated:", forces)
    return forces

def accelerations(bodies):
    accelerations = forces(bodies)
    for i, body in enumerate(bodies):
        accelerations[i] /= body.mass
    #print("Accelerations calculated:", accelerations)
    return accelerations

def normMasses(bodies):
    """Normalize the masses of the bodies to a common scale."""
    total_mass = sum(body.mass for body in bodies)
    for body in bodies:
        body.mass /= total_mass
    print("Normalized masses:", [body.mass for body in bodies])

def solve_velocities(bodies, start, dt, sim_duration=10000):
    """"Use scipy's solve_ivp to evolve the velocities of the bodies."""
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
test_earth = Body("Earth", x, 1, [1, 0], [0, 1])
test_earth2 = Body("Earth2", 5.972e24, 6371e3, [100000], [0])
test_moon = Body("Moon", y, 1, [1.002569, 0], [0, 1.033])
test_sun = Body("Sun", 1.989e30, 1, [0, 0], [0, 0])

earth = Body("Earth", 5.972e24, 6371e3, [0, 0, 0], [0, 0, 0])
earth2 = Body("Earth2", 5.972e24, 6371e3, [100000, 0, 0], [0, 0, 0])
moon = Body("Moon", 7.348e22, 1737e3, [384400e3, 0, 0], [0, 1022, 0])

bodies = [test_earth, test_sun, test_moon]
normMasses(bodies)
#bodies = [earth, moon]

""" evolve = solve_velocities(bodies,0,  0.01, sim_duration=1000000)
print("Evolution result:", evolve)
print(evolve.y.shape)
print("Final positions:", [body.position for body in bodies]) """

fig, ax = plt.subplots()

scat = ax.scatter(0, 0, c="r", s=5, label=f'sun')
scat2 = ax.scatter(0, 0, c="g", s=5, label=f'earth')
scat3 = ax.scatter(0, 0, c="b", s=5, label=f'moon')
line = ax.plot(test_moon.position[0], test_moon.position[1], c="k", alpha=0.2, label=f'earth orbit')[0]
ax.set(xlim=[-2, 2], ylim=[-2, 2], xlabel='X', ylabel='Y')
ax.legend()

def update(frame):
    continuous_evolve = solve_velocities(bodies, 0, 0.01, sim_duration=1000).y
    #print(continuous_evolve[2][-1], continuous_evolve[3][-1])
    #all_evolves += (continuous_evolve)
    # for each frame, update the data stored on each artist.
    x = continuous_evolve[2][-1]
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
    line.set_ydata(np.append(lineData[1], continuous_evolve[1][::100]))
    return (scat, scat2, line)


ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=10)
plt.show()