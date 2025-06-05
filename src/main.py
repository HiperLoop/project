from systems import *

def main():
    """Main function to run the simulation."""
    # Uncomment the desired simulation configuration
    #figure_eight_configureation()
    #figure_eight_configureation_3D()
    #solar_system()
    #solar_system_3D()
    #comet_solar_system()
    system_from_file("2025-06-05_16-01-10.csv", dimension=2) # 2D load test
    #system_from_file("2025-06-05_16-01-10.csv", dimension=3) # 3D load test
    
main()