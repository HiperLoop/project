import numpy as np
from scipy.integrate import solve_ivp
from body import Body
from loaders import *
from measures import *
from measures import Kepler


class Simulation_parameters:
    def __init__(self, **kwargs):
        self.dimension = kwargs.get('dimension', 3)

        self.do_norming = kwargs.get('do_norming', True)
        self.distance_norm = kwargs.get('distance_norm', 149.598023)
        self.velocity_norm = kwargs.get('velocity_norm', 29.8)
        self.gravitational_constant = kwargs.get('gravitational_constant', 1)

        self.step_precision = kwargs.get('step_precision', 100)
        self.step_time = kwargs.get('step_time', 0.01)
        self.step_iterations = kwargs.get('step_iterations', 10000)

class Simulation:
    """Class for simulating gravitational interactions between n bodies."""

    def __init__(self, bodies, parameters, reverse=False, save_to_file=False, auto_run=False):
        """Initialize the starting state and simulation parameters"""
        # ODE calculation variables
        self.current_step = 0
        self.time_step = parameters.step_time  # Time step for the simulation
        self.calculation_step = parameters.step_time/parameters.step_precision

        # get bodies and set their dimension
        self.bodies = bodies
        self.dimension = parameters.dimension
        Body.dimension = parameters.dimension

        # norming parameters
        self.G = parameters.gravitational_constant
        if parameters.do_norming:
            self.norming_distance = parameters.distance_norm
            self.norming_velocity = parameters.velocity_norm
        self.unit_norming = parameters.do_norming  # Whether to normalize units
        self.initial_norming()  # Normalize masses and calculate center of mass

        if reverse: self.reverse_velocities() # reverse velocities to run simulation backwards

        # file saving parameters
        self.save_to_file = save_to_file
        if save_to_file: 
            self.file_name = write_simulation_to_file_init(self.bodies, [parameters.dimension, parameters.step_precision, parameters.step_time, parameters.step_iterations])
            self.file = open(self.file_name, "a")

        # running Kepler calculation
        self.kepler_law_calc = Kepler(num_of_bodies=len(bodies))

        # running without animation
        if auto_run:
            self.iterations = parameters.step_iterations

    def __del__(self):
        """Close the file for writing data if it was opened."""
        if self.save_to_file and hasattr(self, 'file'):
            self.file.close()
            print(f"Simulation data saved to {self.file_name}")

    # region #################################### Accelerations ##############################################
    def twoBody_acceleration(self, body1, body2):
        """Calculate the accelaration fof body one towards body 2."""
        g = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
        r_vector = relative_position(body1, body2)
        distance = np.linalg.norm(r_vector)
        
        if distance <= ((body1.radius + body2.radius)):
            return np.zeros(body1.dimension)  # No force if bodies would be touching, as this sometimes rockets the sun out
        
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
    # endregion #################################### Accelerations ###########################################
    
    # region #################################### Norming ####################################################
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
        """calculate the realtive positions of all bodies with respect to the COM. And update initial vecotr data."""
        for body in self.bodies:
            body.position -= COM_position
            body.initial_vector = body.position
            body.initial_vector_magnitude = np.linalg.norm(body.position)
            body.min_distance = body.initial_vector_magnitude

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
            body.radius /= (1000000 * AU)
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
    # endregion #################################### Norming #################################################
            
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
        self.kepler_law_calc.calculate_Kepler(self.bodies, self.current_step, self.time_step)
        if self.save_to_file: write_simulation_to_file_step(self.file, self.bodies)
        return result

    def reverse_velocities(self):
        """Reverse the velocities of the bodies."""
        for body in self.bodies:
            body.velocity *= -1
    
    # region #################################### Run simulation ##############################################
    def runner(self):
        '''Automatic simulation runner'''
        print("Simulation started")
        while self.current_step < self.iterations:
            self.solve_velocities(0)
            if self.current_step % (self.iterations//1000) == 0:
                print(f'{self.current_step / (self.iterations//100)}% done')
        for body in self.bodies:
            print(f'{body.name} has period: {body.period} and semi-major axis: {body.semimajor_axis}')

    def start(self):
        self.runner()
    # endregion #################################### Run simulation ###########################################