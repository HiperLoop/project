import numpy as np
from scipy.integrate import solve_ivp
from body import Body
from loaders import *
from measures import *

class Simulation:
    """Class for simulating gravitational interactions between n bodies."""

    def __init__(self, bodies, dimension=2, G=1, norming_distance=149.6, norming_velocity=29.8, norm=True, reverse=False, precision = 100, time_step=0.01, save_to_file=False):
        """Initialize the simulation with a list of bodies, their dimensions, and gravitational constant."""
        self.current_step = 0
        self.time_step = time_step  # Time step for the simulation
        self.calculation_step = time_step/precision
        self.bodies = bodies
        self.dimension = dimension
        Body.dimension = dimension  # Set the dimension for the Body class
        self.G = G  # Gravitational constant
        self.norming_distance = norming_distance  # Normalization distance in AU
        self.norming_velocity = norming_velocity
        self.unit_norming = norm  # Whether to normalize units
        self.initial_norming()  # Normalize masses and calculate center of mass
        self.save_to_file = save_to_file  # Whether to save simulation data to a file
        self.initial_vector = relative_position(self.bodies[0], self.bodies[1])
        if reverse: self.reverse_velocities()
        if save_to_file: 
            self.file_name = write_simulation_to_file_init(self.bodies)
            self.file = open(self.file_name, "a")
        self.angles = [0, 0, 0]

    def __del__(self):
        """Close the file if it was opened."""
        if self.save_to_file and hasattr(self, 'file'):
            self.file.close()
            print(f"Simulation data saved to {self.file_name}")

    def twoBody_acceleration(self, body1, body2):
        """Calculate the accelaration fof body one towards body 2."""
        g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        r_vector = relative_position(body1, body2)
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

    def calculate_period(self):
        self.angles[0] = calculate_angle(self.initial_vector, self.bodies)
        if self.angles [0] >= self.angles[1] and self.angles[1] <= self.angles[2]:
            print("=================================================================")
            print(self.current_step * self.time_step)
            print("=================================================================")
        self.angles[2] = self.angles[1]
        self.angles[1] = self.angles[0]

    def solve_velocities(self, start):
        """Solve the equations of motion for the bodies using the Runge-Kutta method."""
        self.current_step += 1
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
        t_span = (0, self.time_step)
        t_evals = np.arange(start, self.time_step, self.calculation_step)
        result = solve_ivp(equations_of_motion, t_span=t_span, t_eval=t_evals, y0=initial_conditions, vectorized=True)
        if self.save_to_file: write_simulation_to_file_step(self.file, self.bodies)
        #if self.save_to_file: write_simulation_to_file_step_y(self.file, result.y)
        #self.calculate_period()
        return result

    def reverse_velocities(self):
        """Reverse the velocities of the bodies."""
        for body in self.bodies:
            body.velocity *= -1