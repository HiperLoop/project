def write_body_to_file():
    """Write a single body's data to the file."""
    name = input("Enter body name: ")
    mass = input("Enter body mass: ")
    diameter = input("Enter body diameter: ")
    dimensions = input("Enter body dimensions (e.g., 2 for 2D, 3 for 3D): ")
    dim_labels = ["X", "Y", "Z"]
    pos = []
    vel = []
    for i in range(int(dimensions)):
        pos.append(input(f"Enter {dim_labels[i]} position in Gm: "))
        vel.append(input(f"Enter {dim_labels[i]} velocity in km/s: "))
    for i in range(int(dimensions), 3):
        pos.append("0")
        vel.append("0")

    file = open('./objects/custom_objects.csv', "a")
    file.write(name + "," + mass + "," + diameter + "," + ",".join(pos) + "," + ",".join(vel) + "\n")

write_body_to_file()