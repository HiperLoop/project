import numpy as np
from scipy.integrate import solve_ivp
from body import Body

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
        print(self.dimension)
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