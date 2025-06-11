from loaders import *
from simulation import Simulation, Simulation_parameters
from animation import Animation, Animation_parameters
from measures import *

class System:
    '''class to store the simulation system'''

    def __init__(self, animate=True, from_file=False, **kwargs):
        animation_params = kwargs.get('animation_parameters', Animation_parameters())
        simulation_params = kwargs.get('simulation_parameters', Simulation_parameters())
        self.animate=animate
        self.name = kwargs.get('name', None)
        if from_file:
            if animate:
                self.animation = Animation(animation_params, data_from_file=True, file_name=kwargs.get('file_name', None))
        else:
            self.sim = Simulation(kwargs.get('bodies', None), simulation_params, reverse=False, save_to_file=kwargs.get('save_to_file', True), auto_run=not animate)
            if animate:
                self.animation = Animation(animation_params, simulation=self.sim)
    
    def run(self):
        if self.animate: self.animation.start()
        else: self.sim.start()

    def reverse(self):
        self.sim.reverse_velocities()
        self.run()

def animate_from_file(file_name, animation_values=Animation_parameters(plot_axis_limits=2, plot_dimension=3, frame_rate=80)):
    sys = System(animate=True, from_file=True, animation_parameters=animation_values, file_name=file_name)
    sys.run()

def solar_system(animate=True, ani_params=Animation_parameters(plot_axis_limits=40, plot_dimension=3, frame_rate=40)):
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_planets('planets.csv')
    # load Sun from the custom CSV file
    sun = load_body_from_custom_csv('custom_objects.csv', dimension=3)[0]
    bodies = [sun] + planets
    solar_system = System(animate=animate, from_file=False, name="Solar system", bodies=bodies, simulation_parameters=Simulation_parameters(dimension=3, do_norming=True, step_time=0.1, precision=10, step_iterations=100000), animation_parameters=ani_params)
    solar_system.run()

def figure_eight(animate=True, ani_params=Animation_parameters(plot_axis_limits=2, plot_dimensions=2, frame_rate=60)):
    system = System(animate=animate, from_file=False, name="Figure eight", bodies=load_body_from_custom_csv('custom_objects.csv', dimension=ani_params.plot_dimension)[-4:-1], simulation_parameters=Simulation_parameters(do_norming=False, dimension=ani_params.plot_dimension, gravitational_constant=3, step_time=0.1), animation_parameters=ani_params)
    system.run()

def system_from_user_input(body_names, **kwargs):
    user_system = System(animate = kwargs.get('animate', False), name="user name", bodies=load_boadies_by_name(body_names, dimension=kwargs.get('dimension', 3)), simulation_parameters=kwargs.get('simulation_parameters', Simulation_parameters()), animation_parameters=kwargs.get('animation_parameters', Animation_parameters()))
    user_system.run()

pre_made_systems = {"Solar system": solar_system, "Figure eight":figure_eight}