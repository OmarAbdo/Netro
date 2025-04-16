"""
TruckFleetSizer Module
------------------------
This module implements the TruckFleetSizer class, which calculates the number of trucks required
to satisfy customer demand based on truck capacity. Two methods are provided:
1. Capacity-based: fleet_size = ceil(total_demand / truck_capacity)
2. Cluster-based: for each customer cluster, compute required trucks = ceil(cluster_demand / truck_capacity),
   and then sum over clusters.

Inspiration and References:
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For classical fleet sizing; see p. 15, lines 5–10.)
    - Mourelo Ferrandez et al., "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm",
      JIEM, p. 377, lines 10–15.
      
This module follows SOLID principles with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import math
from netro.core.entities.vehicle import Truck
from netro.components.customer_clustering.customer_analysis import (
    CustomerAnalyzer,
)  # CustomerAnalyzer as client


class TruckFleetSizer:
    """
    TruckFleetSizer calculates the minimum number of trucks needed.

    Two methods are available:
      a) Capacity-based: fleet_size = ceil(total_demand / truck_capacity)
      b) Cluster-based: for each cluster, required trucks = ceil(cluster_demand / truck_capacity);
         overall fleet size = sum over clusters.

    Citation: Inspired by Solomon (1987, p. 15, lines 5–10) and Mourelo Ferrandez et al. (2016, p. 377, lines 10–15).
    """

    def __init__(self, truck_capacity, orders):
        """
        Initialize the TruckFleetSizer.

        :param truck_capacity: Capacity of a single truck (integer).
        :param orders: List of orders, each order is a dictionary with keys 'id' and 'demand'.
        """
        self.truck_capacity = truck_capacity
        self.orders = orders

    def total_demand(self):
        """
        Compute the total demand from the orders.

        :return: Total demand (integer).
        """
        return sum(order["demand"] for order in self.orders)

    def calculate_capacity_based_fleet_size(self):
        """
        Calculate fleet size using total demand divided by truck capacity.

        :return: Number of trucks = ceil(total_demand / truck_capacity)
        Citation: Solomon (1987, p. 15, lines 5–10).
        """
        total = self.total_demand()
        return math.ceil(total / self.truck_capacity)

    def calculate_cluster_based_fleet_size(self, cluster_assignments):
        """
        Calculate fleet size based on clustering.
        For each cluster, sum the demand and compute:
            required_trucks_cluster = ceil(cluster_demand / truck_capacity)
        Then, fleet size is the sum over clusters.

        :param cluster_assignments: Array-like cluster labels for each order (order list and cluster assignments must align).
        :return: Fleet size (integer) based on clusters.

        Citation: Inspired by Mourelo Ferrandez et al. (2016, p. 377, lines 10–15).
        """
        cluster_demand = {}
        for order, label in zip(self.orders, cluster_assignments):
            cluster_demand[label] = cluster_demand.get(label, 0) + order["demand"]

        fleet_size = 0
        for label, demand in cluster_demand.items():
            # Ignore noise points (label = -1) by treating them as individual outliers
            if label == -1:
                fleet_size += 1
            else:
                fleet_size += math.ceil(demand / self.truck_capacity)
        return fleet_size

    def create_fleet(self, fleet_size):
        """
        Create a list of Truck objects based on the fleet size.

        :param fleet_size: Number of trucks to create.
        :return: List of Truck objects.
        """
        return [Truck(truck_id=i) for i in range(fleet_size)]


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())

    # Path to the Solomon dataset file
    file_path = "dataset/c101.txt"

    # Create an instance of CustomerAnalyzer and load customer data
    from netro.components.customer_clustering.customer_analysis import CustomerAnalyzer

    analyzer = CustomerAnalyzer(file_path)
    customers_df = analyzer.load_data()

    # Convert customer DataFrame to orders (excluding depot)
    orders = CustomerAnalyzer.orders_from_solomon_df(customers_df)

    # Use HDBSCAN for clustering
    labels, clusterer, n_clusters = analyzer.cluster_customers_hdbscan(
        min_cluster_size=5
    )
    print(f"HDBSCAN found {n_clusters} clusters (noise labeled as -1).")

    # Assume each truck has a capacity of 200 units
    truck_capacity = 200

    # Initialize TruckFleetSizer with truck capacity and orders
    fleet_sizer = TruckFleetSizer(truck_capacity, orders)

    # Calculate fleet size using capacity-based method
    capacity_based_size = fleet_sizer.calculate_capacity_based_fleet_size()
    print(f"Capacity-Based Fleet Size: {capacity_based_size} trucks")

    # Calculate fleet size using cluster-based method (using HDBSCAN labels)
    cluster_based_size = fleet_sizer.calculate_cluster_based_fleet_size(labels)
    print(f"Cluster-Based Fleet Size: {cluster_based_size} trucks")

    # Create fleet using the cluster-based fleet size and display details
    print("\nFleet Details (Cluster-Based):")
    fleet = fleet_sizer.create_fleet(cluster_based_size)
    for truck in fleet:
        print(f"Truck ID: {truck.truck_id}, Capacity: {truck.capacity}")
