from loaders import *
from simulation import Simulation, Simualtion_parameters
from animation import Animation, Animation_parameters
from measures import *

class System:
    '''class to store the simulation system'''

    def __init__(self, animate=True, from_file=False, **kwargs):
        animation_params = kwargs.get('animation_parameters', Animation_parameters())
        simulation_params = kwargs.get('simulation_parameters', Simualtion_parameters())
        self.animate=animate
        if from_file:
            if animate:
                self.animation = Animation(animation_params, data_from_file=True, file_name=kwargs.get('file_name', None))
        else:
            self.sim = Simulation(kwargs.get('bodies', None), simulation_params, reverse=False, save_to_file=False, auto_run=not animate)
            if animate:
                self.animation = Animation(animation_params, simulation=self.sim)
    
    def run(self):
        if self.animate: self.animation.start()
        else: self.sim.start()

    def reverse(self):
        self.sim.reverse_velocities()
        self.run()

#test_config = System(animate=True, from_file=False, bodies=load_body_from_custom_csv('custom_objects.csv', dimension=2)[-4:-1], simulation_parameters=Simualtion_parameters(dimension=2, gravitational_constant=3, step_time=0.1), animation_parameters=Animation_parameters(plot_axis_limits=3, plot_dimension=2))
centauri_bodies=load_body_from_custom_csv('centauri.csv', dimension=3)
centauri_system = System(animate=True, from_file=False, bodies=centauri_bodies, simulation_parameters=Simualtion_parameters(dimension=3, do_norming=True, distance_norm=1, step_time=0.1), animation_parameters=Animation_parameters(plot_axis_limits=15, plot_dimension=3, frame_rate=200))


def figure_eight_configureation(animate=True):
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('custom_objects.csv')[-4:-1]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=3, norm=False, reverse=False, time_step=0.1, save_to_file=True, auto_run=not animate, iterations=100)
    # Initialize the animation
    if animate: anim = Animation(simulation=sim, plot_size=6, plot_dimensions=3, frame_rate=100)

def figure_eight_configureation_3D():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('custom_objects.csv', dimension=3)[-4:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=3, norm=False, reverse=False, time_step=0.1, save_to_file=False)
    # Initialize the animation
    anim = Animation(simulation=sim, plot_size=6, plot_dimensions=1.5, frame_rate=100)

def solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('custom_objects.csv')[0]
    bodies = [sun] + planets
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149.6, norming_velocity=29.8, norm=True, reverse=False, precision=1000, time_step=0.1, save_to_file=False)
    # Initialize the animation
    anim = Animation(simulation=sim, plot_size=6, plot_dimensions=40, frame_rate=100)

def comet_solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('custom_objects.csv')[0]
    # Load comet from the custom CSV file
    comet = load_body_from_custom_csv('custom_objects.csv')[-1]
    earth = load_body_from_csv('planets.csv', dimension=2)[3]  # Load Earth for reference
    bodies = [sun] + [earth]# + planets

    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149.6, norming_velocity=29.8, norm=True, reverse=False, precision=10, time_step=0.01, save_to_file=True)
    # Initialize the animation
    anim = Animation(simulation=sim, plot_size=6, plot_dimensions=3, frame_rate=200)

def system_from_file(file_name, dimension):
    anim = Animation(plot_size=6, plot_dimensions=3, frame_rate=200, data_from_file=True, dimension=dimension, file_name=file_name)

def solar_system_3D():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('planets.csv', dimension=3)
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('custom_objects.csv', dimension=3)[0]
    bodies = [sun] + planets
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(simulation=sim, plot_size=6, plot_dimensions=40, frame_rate=100)