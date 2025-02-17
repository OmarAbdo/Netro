"""
TruckFleetSizer Module
------------------------
This module implements the TruckFleetSizer class, which calculates the number of trucks required
to satisfy customer demand based on truck capacity. Two methods are provided:
1. A capacity-based heuristic: fleet_size = ceil(total_demand / truck_capacity)
2. A cluster-based heuristic: for each customer cluster, compute required trucks = ceil(cluster_demand / truck_capacity)
   and then sum over clusters.

Inspiration and References:
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For classical fleet sizing; see p. 15, lines 5–10.)
    - Mourelo Ferrandez et al., "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm",
      JIEM, p. 377, lines 10–15, for clustering-based ideas.
      
This module follows SOLID principles with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import math
from entities.truck import Truck
from components.customer_analysis.customer_analysis import (
    CustomerAnalyzer,
)  # CustomerAnalyzer as client


class TruckFleetSizer:
    """
    TruckFleetSizer calculates the minimum number of trucks needed.

    Two methods are available:
      a) Capacity-based: fleet_size = ceil(total_demand / truck_capacity)
      b) Cluster-based: for each cluster, required trucks = ceil(cluster_demand / truck_capacity);
         overall fleet size = sum over clusters.

    Note: In a traditional truck-only system, routing inefficiencies may force additional trucks.
    In a multi-echelon system where trucks act as mobile depots, clustering can help refine fleet sizing.
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

    def calculate_capacity_based_fleet_size(self):
        """
        Calculate fleet size using total demand / truck capacity.

        :return: Number of trucks (integer) = ceil(total_demand / truck_capacity)
        Citation: Inspired by Solomon (1987, p. 15, lines 5–10).
        """
        total = self.total_demand()
        return math.ceil(total / self.truck_capacity)

    def calculate_cluster_based_fleet_size(self, cluster_assignments):
        """
        Calculate fleet size based on clustering.
        For each cluster, sum the demand and compute:
            required_trucks_cluster = ceil(cluster_demand / truck_capacity)
        Then, fleet size is the sum of required trucks for all clusters.

        :param cluster_assignments: List or array of cluster labels corresponding to each order.
                                    Must be in the same order as self.orders.
        :return: Fleet size (integer) based on clusters.

        Citation: This approach is inspired by clustering-based methods (Mourelo Ferrandez et al. (2016), p. 377, lines 10–15).
        """
        # Build a mapping from cluster label to total demand
        cluster_demand = {}
        # Assume orders and cluster_assignments are aligned and have the same length.
        for order, label in zip(self.orders, cluster_assignments):
            cluster_demand[label] = cluster_demand.get(label, 0) + order["demand"]

        fleet_size = 0
        for label, demand in cluster_demand.items():
            trucks_for_cluster = math.ceil(demand / self.truck_capacity)
            fleet_size += trucks_for_cluster
        return fleet_size

    def create_fleet(self, fleet_size):
        """
        Create a list of Truck objects corresponding to the given fleet size.

        :param fleet_size: Number of trucks to create.
        :return: List of Truck objects.
        """
        return [Truck(truck_id=i) for i in range(fleet_size)]


if __name__ == "__main__":
    # Usage Example for TruckFleetSizer using the actual CustomerAnalyzer from the Solomon dataset.
    import sys
    import os

    sys.path.append(os.getcwd())

    # Path to the Solomon dataset file
    file_path = "dataset/c101.txt"

    # Create an instance of CustomerAnalyzer and load customer data
    analyzer = CustomerAnalyzer(file_path)
    customers_df = analyzer.load_data()

    # Convert customer DataFrame into orders (excluding depot) using the static method from CustomerAnalyzer
    orders = CustomerAnalyzer.orders_from_solomon_df(customers_df)

    # Assume each truck has a capacity of 200 units
    truck_capacity = 200

    # Initialize the TruckFleetSizer with the truck capacity and orders
    fleet_sizer = TruckFleetSizer(truck_capacity, orders)

    # Calculate fleet size using the capacity-based method
    capacity_based_size = fleet_sizer.calculate_capacity_based_fleet_size()
    print(f"Capacity-Based Fleet Size: {capacity_based_size} trucks")

    # Now, perform clustering using the CustomerAnalyzer to get cluster assignments
    n_clusters = 5  # For example, cluster customers into 5 groups
    labels, centroids = analyzer.cluster_customers(n_clusters)

    # Calculate fleet size using the cluster-based method
    cluster_based_size = fleet_sizer.calculate_cluster_based_fleet_size(labels)
    print(f"Cluster-Based Fleet Size: {cluster_based_size} trucks")

    # Create fleets accordingly (you can choose one method's result)
    print("\nFleet Details (Cluster-Based):")
    fleet = fleet_sizer.create_fleet(cluster_based_size)
    for truck in fleet:
        print(f"Truck ID: {truck.truck_id}, Capacity: {truck.capacity}")
