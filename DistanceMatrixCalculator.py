import pandas as pd
import numpy as np


class DistanceMatrixCalculator:
    """
    A class to handle loading, preprocessing, and analyzing Solomon VRPTW instances for vehicle routing problems.
    in order to compute the distance matrix.
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

        - Skips any non-numeric or empty lines.
        - Extracts vehicle and depot information.
        - Parses customer data, including coordinates, demands, time windows, and service times.
        """
        with open(self.file_path, "r") as file:
            lines = file.readlines()

        # Skip header lines and find the numeric line with vehicle and capacity info
        for line in lines:
            if (
                line.strip() and line.strip().split()[0].isdigit()
            ):  # Check for non-empty and numeric content
                header = line.strip().split()
                self.num_vehicles, self.vehicle_capacity = int(header[0]), int(header[1])
                break

        # Initialize a list to store customer data
        customer_data = []

        # Parse remaining lines for customer details
        for line in lines[lines.index(line) + 1 :]:
            parts = line.strip().split()
            if len(parts) == 7:  # Check for valid customer/depot entry
                customer_data.append(
                    {
                        "ID": int(parts[0]),  # Unique identifier for the customer or depot
                        "X": float(parts[1]),  # X-coordinate of the location
                        "Y": float(parts[2]),  # Y-coordinate of the location
                        "Demand": int(parts[3]),  # Delivery demand at this location
                        "Ready_Time": int(parts[4]),  # Earliest time delivery can start
                        "Due_Time": int(parts[5]),  # Latest time delivery must be completed
                        "Service_Time": int(
                            parts[6]
                        ),  # Time required to serve the customer
                    }
                )

        # Store customer data in a Pandas DataFrame
        self.customers = pd.DataFrame(customer_data)

    def relax_time_windows(self, max_due_time=1000):
        """
        Relax the time windows for simulation purposes.

        - Sets all Ready_Time to 0, allowing deliveries to start at any time.
        - Extends Due_Time to a maximum value to simulate relaxed delivery schedules.

        :param max_due_time: Maximum due time for all deliveries (default: 1000).
        """
        self.customers["Ready_Time"] = 0
        self.customers["Due_Time"] = max_due_time

    def compute_distance_matrix(self):
        """
        Compute the distance matrix between all customer locations and the depot.

        - Uses Euclidean distance to calculate pairwise distances.
        - Returns a 2D NumPy array where entry (i, j) represents the distance from location i to location j.

        :return: A NumPy array representing the distance matrix.
        """
        num_locations = len(self.customers)  # Total number of locations
        distance_matrix = np.zeros(
            (num_locations, num_locations)
        )  # Initialize the matrix with zeros

        # Compute distances between each pair of locations
        for i, loc1 in self.customers.iterrows():
            for j, loc2 in self.customers.iterrows():
                distance_matrix[i, j] = np.sqrt(
                    (loc1["X"] - loc2["X"]) ** 2 + (loc1["Y"] - loc2["Y"]) ** 2
                )

        return distance_matrix


# Example usage
if __name__ == "__main__":
    # Initialize the dataset with the path to the Solomon VRPTW instance file
    distanceMatrixCalculator = DistanceMatrixCalculator("dataset/c101.txt")

    # Load data from the file
    distanceMatrixCalculator.load_data()

    # Relax the time windows for simulation purposes
    distanceMatrixCalculator.relax_time_windows()

    # Compute the distance matrix for all locations
    distance_matrix = distanceMatrixCalculator.compute_distance_matrix()

    # Display the first few rows of customer data
    print(distanceMatrixCalculator.customers.head())
