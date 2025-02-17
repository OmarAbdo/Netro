"""
TruckRouter Module
--------------------
This module integrates customer analysis and truck routing to plan routes for trucks 
acting as mobile depots that visit clusters of customers. It performs the following:
    1. Loads customer data and performs clustering using CustomerAnalyzer.
    2. Computes cluster centroids and aggregate demand for each cluster.
    3. Builds a distance matrix for the cluster centroids.
    4. Uses OR‑Tools to solve a Vehicle Routing Problem (VRP) at the cluster level.
    5. Visualizes the truck routes over the map of clusters.

Inspiration and References:
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm", JIEM, p. 377 (lines 10–15) for clustering.
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints", p. 15 (lines 5–10) for fleet sizing and VRP formulation.
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E, p. 67 (lines 15–20) for local search improvements.
    
This module follows SOLID principles with one class per file.
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from components.customer_clustering.customer_analysis import (
    CustomerAnalyzer,
)  # Our customer analysis module
from sklearn.cluster import KMeans


class TruckRouter:
    """
    TruckRouter integrates customer analysis and truck routing to determine optimal truck routes
    through cluster centroids, which act as strategic launch points for autonomous robots.
    """

    def __init__(self, file_path, n_clusters, truck_capacity):
        """
        Initialize TruckRouter.

        :param file_path: Path to the Solomon dataset file.
        :param n_clusters: Number of clusters to form.
        :param truck_capacity: Capacity of a single truck (used to filter or split clusters if necessary).
        """
        self.file_path = file_path
        self.n_clusters = n_clusters
        self.truck_capacity = truck_capacity
        self.customers_df = None
        self.cluster_labels = None
        self.centroids = None
        self.cluster_demands = None
        self.depot = (
            None  # Depot coordinates (from the Solomon dataset, row where ID == 0)
        )

    def load_and_cluster(self):
        """
        Loads customer data using CustomerAnalyzer, performs clustering (ignoring time windows),
        computes centroids and cluster demands.

        :return: Tuple (centroids, cluster_demands)
        """
        # Instantiate and load customer data
        analyzer = CustomerAnalyzer(self.file_path)
        self.customers_df = analyzer.load_data()

        # Set depot as the first row (ID==0)
        depot_row = self.customers_df[self.customers_df["ID"] == 0].iloc[0]
        self.depot = np.array([depot_row["X"], depot_row["Y"]])

        # Exclude depot for clustering
        customers = self.customers_df[self.customers_df["ID"] != 0].copy()
        coords = customers[["X", "Y"]].values

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(coords)
        self.centroids = kmeans.cluster_centers_

        # Compute aggregate demand for each cluster
        customers["Cluster"] = self.cluster_labels
        self.cluster_demands = customers.groupby("Cluster")["Demand"].sum().to_dict()

        return self.centroids, self.cluster_demands

    def compute_cluster_distance_matrix(self):
        """
        Compute a distance matrix among all cluster centroids.

        :return: A 2D numpy array (n_clusters x n_clusters) with Euclidean distances.
        """
        num_clusters = self.n_clusters
        dist_matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(num_clusters):
                dist_matrix[i, j] = np.linalg.norm(
                    self.centroids[i] - self.centroids[j]
                )
        return dist_matrix

    def create_vrp_data_model(self, distance_matrix):
        """
        Create data model for the VRP using cluster centroids.
        The depot is the first node (we insert the depot as node 0).

        :param distance_matrix: Distance matrix among cluster centroids.
        :return: A dictionary containing VRP data.
        """
        # Insert the depot at index 0. The new list of nodes: depot, then clusters.
        nodes = np.vstack([self.depot, self.centroids])
        num_nodes = len(nodes)
        # Recompute distance matrix with depot included
        full_distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                full_distance_matrix[i, j] = np.linalg.norm(nodes[i] - nodes[j])

        # Demands: depot demand = 0, then each cluster's demand (if a cluster demand exceeds truck capacity,
        # one may need to split the cluster; here we assume clusters are feasible for now)
        demands = [0]
        for i in range(self.n_clusters):
            demands.append(self.cluster_demands.get(i, 0))

        data = {
            "distance_matrix": full_distance_matrix.astype(
                int
            ).tolist(),  # OR-Tools expects integer distances
            "demands": demands,
            "vehicle_capacities": [
                self.truck_capacity
            ],  # We'll solve for one truck route, then later extend to multiple trucks if needed
            "num_vehicles": 1,
            "depot": 0,
        }
        return data

    def solve_cluster_vrp(self, data):
        """
        Solve the VRP for clusters using OR-Tools.

        :param data: VRP data model.
        :return: A dictionary with the route and total distance.
        """
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # no slack
            data["vehicle_capacities"],
            True,
            "Capacity",
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            return self.extract_solution(manager, routing, solution)
        else:
            return {"route": None, "total_distance": None}

    def extract_solution(self, manager, routing, solution):
        """
        Extract the route from the OR-Tools solution.

        :return: Dictionary with 'route' (list of node indices) and 'total_distance'.
        """
        index = routing.Start(0)
        route = []
        total_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        route.append(manager.IndexToNode(index))
        return {"route": route, "total_distance": total_distance}

    def plot_truck_route(self, vrp_solution):
        """
        Plot the truck route over the cluster centroids and depot.
        """
        # Nodes: first is depot, then centroids
        nodes = np.vstack([self.depot, self.centroids])
        route = vrp_solution["route"]

        plt.figure(figsize=(8, 8))
        plt.scatter(nodes[0, 0], nodes[0, 1], c="red", marker="s", s=100, label="Depot")
        plt.scatter(nodes[1:, 0], nodes[1:, 1], c="blue", label="Cluster Centroids")
        for i, (x, y) in enumerate(nodes[1:], start=1):
            plt.text(x + 0.5, y + 0.5, f"C{i-1}", fontsize=9)
        # Draw route lines
        route_coords = nodes[route]
        plt.plot(route_coords[:, 0], route_coords[:, 1], "k--", label="Truck Route")
        plt.title("Truck Route over Cluster Centroids")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Usage Example for TruckRouter
    import sys
    import os

    sys.path.append(os.getcwd())

    # Path to Solomon dataset
    file_path = "dataset/c101.txt"

    # Define number of clusters and truck capacity (assume truck capacity in same units as order demand)
    n_clusters = 5
    truck_capacity = 200

    # Create instance of TruckRouter and perform analysis
    router = TruckRouter(file_path, n_clusters, truck_capacity)
    centroids, cluster_demands = router.load_and_cluster()

    print("Cluster Centroids:")
    print(centroids)
    print("Cluster Demands:")
    print(cluster_demands)

    # Compute distance matrix among clusters (centroids)
    cluster_distance_matrix = router.compute_cluster_distance_matrix()

    # Create VRP data model for clusters (including depot)
    vrp_data = router.create_vrp_data_model(cluster_distance_matrix)

    # Solve the VRP for clusters using OR-Tools
    vrp_solution = router.solve_cluster_vrp(vrp_data)

    print("\nTruck Route (Indices with 0 as depot, then clusters):")
    print(vrp_solution["route"])
    print(f"Total Truck Route Distance: {vrp_solution['total_distance']}")

    # Plot the truck route over the clusters
    router.plot_truck_route(vrp_solution)
