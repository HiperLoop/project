from loaders import *
from simulation import Simulation
from animation import Animation
from measures import *

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
    sim = Simulation(bodies, dimension=3, G=3, norm=False, reverse=False, time_step=0.1, save_to_file=True)
    # Initialize the animation
    anim = Animation(simulation=sim, plot_size=6, plot_dimensions=3, frame_rate=100)

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