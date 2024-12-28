import pandas as pd

# Step 1: Load and Parse Data
# We’ll need to load these files into a simulation-friendly format (e.g., Python dictionaries or dataframes).


def load_solomon_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse header for vehicle capacity
    vehicle_info = lines[0].split()
    num_vehicles, vehicle_capacity = int(vehicle_info[0]), int(vehicle_info[1])
    
    # Parse customer and depot data
    data = []
    for line in lines[1:]:
        split_line = line.split()
        if len(split_line) == 7:  # Valid customer or depot entry
            customer = {
                "ID": int(split_line[0]),
                "X": float(split_line[1]),
                "Y": float(split_line[2]),
                "Demand": int(split_line[3]),
                "Ready_Time": int(split_line[4]),
                "Due_Time": int(split_line[5]),
                "Service_Time": int(split_line[6]),
            }
            data.append(customer)
    
    return num_vehicles, vehicle_capacity, pd.DataFrame(data)

# Example usage
file_path = "path/to/C101.txt"
num_vehicles, vehicle_capacity, customers = load_solomon_instance(file_path)
print(customers.head())

