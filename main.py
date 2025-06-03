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
    G = 12000  # Modified gravitational constant
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
    print("Forces calculated:", forces)
    return forces

def accelerations(bodies):
    accelerations = forces(bodies)
    for i, body in enumerate(bodies):
        accelerations[i] /= body.mass
    print("Accelerations calculated:", accelerations)
    return accelerations

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
    t_eval = np.linspace(t_span[0], t_span[1], sim_duration)
    result = solve_ivp(equations_of_motion, t_span, initial_conditions,t_eval=t_eval, vectorized=True)
    return result

test_earth = Body("Earth", 1, 0.01657388137, [0, 0], [0, 0])
test_earth2 = Body("Earth2", 5.972e24, 6371e3, [100000], [0])
test_moon = Body("Moon", 0.01230408573, 0.00451873048, [1, 0], [0, 100])

earth = Body("Earth", 5.972e24, 6371e3, [0, 0, 0], [0, 0, 0])
earth2 = Body("Earth2", 5.972e24, 6371e3, [100000, 0, 0], [0, 0, 0])
moon = Body("Moon", 7.348e22, 1737e3, [384400e3, 0, 0], [0, 1022, 0])

bodies = [test_earth, test_moon]
#bodies = [earth, moon]

evolve = solve_velocities(bodies,0,  0.01, sim_duration=1000000)
print("Evolution result:", evolve)
print(evolve.y.shape)
print("Final positions:", [body.position for body in bodies])

""" plt.figure(figsize=(14, 7))
for i in [0, 2]:
    plt.subplot(1, 2, 1)
    plt.plot(evolve.y[i], evolve.y[i+1], label='%s position' % bodies[(i%4)//2].name)
    
plt.legend()

for i in [4, 6]:
    plt.subplot(1, 2, 2)
    plt.plot(evolve.y[i], evolve.y[i+1], label='%s velocity' % bodies[(i%4)//2].name)

plt.legend()
plt.show() """


fig, ax = plt.subplots()

scat = ax.scatter(0, 0, c="r", s=5, label=f'moon')
scat2 = ax.scatter(0, 0, c="b", s=5, label=f'earth')
ax.set(xlim=[-2, 2], ylim=[-2, 2], xlabel='X', ylabel='Y')
ax.legend()

continuous_evolve = [0]
continuous_evolve = solve_velocities(bodies, 0, 0.001, sim_duration=1000).y
def update(frame):
    continuous_evolve = solve_velocities(bodies, 0, 0.001, sim_duration=1000).y
    # for each frame, update the data stored on each artist.
    x = continuous_evolve[2][::1000]
    y = continuous_evolve[3][::1000]
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:
    # for each frame, update the data stored on each artist.
    x2 = continuous_evolve[0][::1000]
    y2 = continuous_evolve[1][::1000]
    # update the scatter plot:
    data2 = np.stack([x2, y2]).T
    scat2.set_offsets(data2)
    # update the line plot:
    return (scat, scat2)


ani = animation.FuncAnimation(fig=fig, func=update, frames=4000, interval=10)
plt.show()