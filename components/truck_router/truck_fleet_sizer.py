"""
TruckFleetSizer Module
------------------------
This module implements the TruckFleetSizer class, which calculates the number of trucks required
to satisfy customer demand based on truck capacity. The calculation uses a simple heuristic:
divide the total demand by the truck capacity and round up.

Inspiration and References:
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For classical fleet sizing methods; see p. 15, lines 5–10.)
    - Mourelo Ferrandez et al., "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm",
      JIEM, p. 377, lines 10–15.

This module follows SOLID principles with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import math
from entities.truck import Truck
from components.customer_analysis.customer_analysis import (
    CustomerAnalyzer,
)  # Import the actual customer analysis module


class TruckFleetSizer:
    """
    TruckFleetSizer calculates the minimum number of trucks needed based on total customer demand
    and a given truck capacity.

    The calculation uses:
        number_of_trucks = ceil(total_demand / truck_capacity)

    This heuristic is standard in VRP literature (see Solomon, 1987, p. 15, lines 5–10).
    """

    def __init__(self, truck_capacity, orders):
        """
        Initialize the TruckFleetSizer.

        :param truck_capacity: Capacity of a single truck (integer).
        :param orders: List of orders, each order is a dictionary containing at least:
                       - 'id': order identifier
                       - 'demand': numeric demand of the order
        """
        self.truck_capacity = truck_capacity
        self.orders = orders

    def total_demand(self):
        """
        Compute the total demand from the list of orders.

        :return: Total demand as an integer.
        """
        return sum(order["demand"] for order in self.orders)

    def calculate_fleet_size(self):
        """
        Calculate the minimum number of trucks required to satisfy the total demand.

        :return: The number of trucks (integer), computed as:
                 ceil(total_demand / truck_capacity)

        Citation: Inspired by Solomon (1987, p. 15, lines 5–10).
        """
        total = self.total_demand()
        return math.ceil(total / self.truck_capacity)

    def create_fleet(self):
        """
        Create a list of Truck objects corresponding to the calculated fleet size.

        :return: List of Truck objects.
        """
        fleet_size = self.calculate_fleet_size()
        fleet = [Truck(truck_id=i) for i in range(fleet_size)]
        return fleet


if __name__ == "__main__":
    # Usage Example for TruckFleetSizer using the actual CustomerAnalyzer from the Solomon dataset.
    # Ensure the current working directory is appended so that module imports work smoothly.
    import sys
    import os

    sys.path.append(os.getcwd())

    # Path to the Solomon dataset file
    file_path = "dataset/c101.txt"

    # Create an instance of CustomerAnalyzer (from our customer_analysis module)
    from components.customer_analysis.customer_analysis import CustomerAnalyzer

    analyzer = CustomerAnalyzer(file_path)

    # Load the customer data and convert it into orders (excluding depot)
    customers_df = analyzer.load_data()
    orders = analyzer.orders_from_solomon_df(customers_df)

    # For this example, assume each truck has a capacity of 200 (units)
    truck_capacity = 200

    # Initialize the TruckFleetSizer with the truck capacity and the orders obtained from customer analysis
    fleet_sizer = TruckFleetSizer(truck_capacity, orders)

    # Calculate total demand and required fleet size based on the orders from the Solomon dataset
    total_demand = fleet_sizer.total_demand()
    num_trucks = fleet_sizer.calculate_fleet_size()

    print(f"Total Demand (from customers, excluding depot): {total_demand}")
    print(f"Calculated Fleet Size: {num_trucks} trucks")

    # Create the fleet (list of Truck objects)
    fleet = fleet_sizer.create_fleet()
    print("\nFleet Details:")
    for truck in fleet:
        print(f"Truck ID: {truck.truck_id}, Capacity: {truck.capacity}")
