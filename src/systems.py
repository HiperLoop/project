from loaders import *
from simulation import Simulation
from animation import Animation
from measures import *

def figure_eight_configureation():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('./objects/custom_objects.csv')[-3:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=3, norm=False, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=3, sim_duration=1000, frame_rate=100)

def figure_eight_configureation_3D():
    """Configure the simulation for the figure-eight configuration."""
    bodies = load_body_from_custom_csv('./objects/custom_objects.csv', dimension=3)[-4:]
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=3, norm=False, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=3, sim_duration=1000, frame_rate=100)

def solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('./objects/planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('./objects/custom_objects.csv')[0]
    bodies = [sun] + planets
    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149.6, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=40, sim_duration=1000, frame_rate=100)

def comet_solar_system():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('./objects/planets.csv')
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('./objects/custom_objects.csv')[0]
    # Load comet from the custom CSV file
    comet = load_body_from_custom_csv('./objects/custom_objects.csv')[-1]
    earth = load_body_from_csv('./objects/planets.csv', dimension=2)[3]  # Load Earth for reference
    bodies = [sun] + [earth]# + planets

    # Initialize the simulation
    sim = Simulation(bodies, dimension=2, G=1, norming_distance=149.6, norming_velocity=29.8, norm=True, reverse=False, time_step=0.01, save_to_file=False)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=15, sim_duration=1000, frame_rate=2000)

def solar_system_3D():
    """Configure the simulation for the solar system."""
    # Load planets from the CSV file
    planets = load_body_from_csv('./objects/planets.csv', dimension=3)
    #load Sun from the custom CSV file
    sun = load_body_from_custom_csv('./objects/custom_objects.csv', dimension=3)[0]
    bodies = [sun] + planets
    # Initialize the simulation
    sim = Simulation(bodies, dimension=3, G=1, norming_distance=149, norming_velocity=29.8, norm=True, reverse=False, time_step=0.1)
    # Initialize the animation
    anim = Animation(sim, plot_size=6, plot_dimensions=40, sim_duration=1000, frame_rate=100)