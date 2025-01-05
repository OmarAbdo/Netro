# DistanceMatrixCalculator.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import time


class DistanceMatrixCalculator:
    """
    A class to handle loading, preprocessing, and analyzing Solomon VRPTW instances for vehicle routing problems.
    """

    def __init__(self, file_path):
        """
        Initialize the DistanceMatrixCalculator class with the file path.

        :param file_path: Path to the Solomon instance text file.
        """
        self.file_path = file_path
        self.num_vehicles = 0  # Number of vehicles available for routing
        self.vehicle_capacity = 0  # Capacity of each vehicle
        self.customers = None  # DataFrame to store customer information

    def load_data(self):
        """
        Load and parse the Solomon VRPTW dataset from the file.

        - Reads the file line by line.
        - Extracts vehicle and depot information.
        - Parses customer data, including coordinates, demands, time windows, and service times.
        """
        print("Loading data from the file...")
        start_time = time.time()

        with open(self.file_path, "r") as file:
            lines = file.readlines()

        # Find the line with vehicle number and capacity
        vehicle_info_found = False
        customer_data_start_idx = 0

        for idx, line in enumerate(lines):
            if line.strip().startswith("NUMBER"):
                # The next line contains the numbers
                number_line = lines[idx + 1]
                parts = number_line.strip().split()
                if len(parts) >= 2:
                    self.num_vehicles = int(parts[0])
                    self.vehicle_capacity = int(parts[1])
                    vehicle_info_found = True
                    customer_data_start_idx = idx + 2  # Start after the vehicle info
                    break

        if not vehicle_info_found:
            raise ValueError("Vehicle information not found in the dataset.")

        # Find the header line for customer data
        for idx in range(customer_data_start_idx, len(lines)):
            if lines[idx].strip().startswith("CUST NO"):
                customer_data_start_idx = idx + 1  # Data starts after this line
                break

        # Initialize a list to store customer data
        customer_data = []

        # Parse remaining lines for customer details
        for line in tqdm(lines[customer_data_start_idx:], desc="Parsing customer data"):
            parts = line.strip().split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    customer_data.append(
                        {
                            "ID": int(parts[0]),
                            "X": float(parts[1]),
                            "Y": float(parts[2]),
                            "Demand": int(parts[3]),
                            "Ready_Time": int(parts[4]),
                            "Due_Time": int(parts[5]),
                            "Service_Time": int(parts[6]),
                        }
                    )
                except ValueError as ve:
                    print(f"Skipping line due to parsing error: {line.strip()}")
                    continue  # Skip lines that don't conform to expected format

        # Store customer data in a Pandas DataFrame
        self.customers = pd.DataFrame(customer_data)
        end_time = time.time()
        print(f"Data loaded successfully in {end_time - start_time:.2f} seconds.")

    def compute_distance_matrix(self):
        """
        Compute the distance matrix between all customer locations and the depot.

        - Uses Euclidean distance to calculate pairwise distances.
        - Returns a 2D NumPy array where entry (i, j) represents the distance from location i to location j.

        :return: A NumPy array representing the distance matrix.
        """
        print("Computing the distance matrix...")
        start_time = time.time()

        # Extract coordinates as a NumPy array
        coords = self.customers[["X", "Y"]].values
        num_locations = len(coords)

        # Vectorized computation of Euclidean distances
        distance_matrix = np.sqrt(
            np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2)
        )

        end_time = time.time()
        print(f"Distance matrix computed in {end_time - start_time:.2f} seconds.")
        return distance_matrix

    def get_demands(self):
        """Get the demands for each customer."""
        return self.customers["Demand"].tolist()

    def get_ready_times(self):
        """Get the ready times for each customer."""
        return self.customers["Ready_Time"].tolist()

    def get_due_times(self):
        """Get the due times for each customer."""
        return self.customers["Due_Time"].tolist()

    def get_service_times(self):
        """Get the service times for each customer."""
        return self.customers["Service_Time"].tolist()


# Usage Example
if __name__ == "__main__":
    # Path to the Solomon VRPTW instance file
    file_path = "dataset/c101.txt"

    # Initialize the DistanceMatrixCalculator
    calculator = DistanceMatrixCalculator(file_path)

    # Load the dataset
    calculator.load_data()

    # Compute the distance matrix
    distance_matrix = calculator.compute_distance_matrix()

    # Extract specific data
    demands = calculator.get_demands()
    ready_times = calculator.get_ready_times()
    due_times = calculator.get_due_times()
    service_times = calculator.get_service_times()

    # Display extracted data
    print("\n--- Extracted Data ---")
    print(f"Number of Customers: {len(demands) - 1} (excluding depot)")
    print("Demands:", demands)
    print("Ready Times:", ready_times)
    print("Due Times:", due_times)
    print("Service Times:", service_times)

    # Display a snippet of the distance matrix
    print("\n--- Distance Matrix (First 5x5) ---")
    print(distance_matrix[:5, :5])  # Display the first 5x5 portion of the matrix
