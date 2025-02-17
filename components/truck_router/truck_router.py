"""
TruckRouter Module
--------------------
This module integrates customer analysis and truck routing to plan routes for trucks acting as mobile depots.
It performs the following:
    1. Loads and clusters customer data using HDBSCAN (via CustomerAnalyzer).
    2. Applies capacity-aware splitting (via CapacitatedClusterSplitter) so that each (sub)cluster's demand is ≤ truck capacity.
    3. Computes cluster centroids and capacity-adjusted demands.
    4. Solves a TSP (using OR-Tools) on the depot and cluster centroids.
    5. Splits the TSP route into multiple truck trips based on the truck's capacity.
    6. Visualizes the final truck trips on a 2D map.

Inspiration and References:
    - McInnes et al. (2017), "HDBSCAN: Hierarchical Density Based Clustering", JOSS, p. 2, lines 20–30.
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm", JIEM, p. 377, lines 10–15.
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints", p. 15, lines 5–10.
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E, p. 67, lines 15–20.

This module is implemented in an OOP style with one class per file.
"""

import sys
import os
sys.path.append(os.getcwd())

from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from components.customer_clustering.customer_analysis import CustomerAnalyzer
from components.customer_clustering.capacitated_clustering import CapacitatedClusterSplitter

class TruckRouter:
    """
    TruckRouter plans the truck routing as a multi-trip VRP over clusters/subclusters.
    The truck acts as a mobile depot that, for each trip, loads orders from the main depot,
    visits a sequence of clusters (ensuring cumulative demand ≤ truck capacity), and returns to reload.
    """
    def __init__(self, file_path, min_cluster_size, truck_capacity):
        """
        Initialize TruckRouter.

        :param file_path: Path to the Solomon dataset.
        :param min_cluster_size: Minimum cluster size for HDBSCAN.
        :param truck_capacity: Capacity of the truck.
        """
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity

    def load_cluster_data(self):
        """
        Loads customer data, performs HDBSCAN clustering, then applies capacity-aware splitting.
        
        :return: A tuple (depot, final_labels, final_centroids, cluster_demands) where:
                 - depot: [X, Y] coordinates of the depot.
                 - final_labels: New cluster labels after splitting.
                 - final_centroids: Coordinates of the (sub)cluster centroids.
                 - cluster_demands: Dictionary mapping new cluster label to its total demand.
        """
        # Load customer data using CustomerAnalyzer
        analyzer = CustomerAnalyzer(self.file_path)
        customers_df = analyzer.load_data()

        # Set depot as the first row (assumed ID==0)
        depot_row = customers_df[customers_df["ID"] == 0].iloc[0]
        depot = np.array([depot_row["X"], depot_row["Y"]])

        # Cluster customers using HDBSCAN (excluding depot)
        df_customers = customers_df[customers_df["ID"] != 0].copy()
        coords = df_customers[["X", "Y"]].values
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        initial_labels = clusterer.fit_predict(coords)
        print(f"HDBSCAN initially found {len(set(initial_labels)) - (1 if -1 in initial_labels else 0)} clusters (noise labeled as -1).")

        # Apply capacitated splitting using CapacitatedClusterSplitter
        splitter = CapacitatedClusterSplitter(df_customers, initial_labels, self.truck_capacity)
        final_labels, final_centroids, subcluster_mapping, new_cluster_demands = splitter.split_clusters()

        print("Cluster Demands after Splitting:")
        print(new_cluster_demands)
        return depot, final_labels, final_centroids, new_cluster_demands

    def compute_distance_matrix(self, nodes):
        """
        Compute Euclidean distance matrix for given nodes.
        
        :param nodes: 2D numpy array of coordinates.
        :return: A 2D numpy array (len(nodes) x len(nodes)) of distances.
        """
        num_nodes = len(nodes)
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist_matrix[i, j] = np.linalg.norm(nodes[i] - nodes[j])
        return dist_matrix

    def solve_tsp(self, distance_matrix):
        """
        Solve the TSP for the given distance matrix using OR-Tools.
        
        :param distance_matrix: 2D list or numpy array of distances.
        :return: A list representing the TSP route (list of node indices).
        """
        num_nodes = len(distance_matrix)
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            return None

        # Extract route
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route

    def split_tsp_route_into_trips(self, tsp_route, demands):
        """
        Given a TSP route (list of node indices) and a list of demands for each node (first element is depot demand = 0),
        split the route into multiple trips. For each trip, starting from the depot, accumulate demand until the next node
        would exceed truck capacity. Then, end the trip by returning to the depot, and continue.
        
        :param tsp_route: List of node indices (with depot as node 0).
        :param demands: List of demands corresponding to nodes in the VRP data model.
        :return: List of trips; each trip is a list of node indices representing a single route (starting and ending at depot).
        """
        trips = []
        current_trip = [0]  # start at depot
        current_load = 0
        # Exclude depot at beginning and end of tsp_route
        for node in tsp_route[1:-1]:
            if current_load + demands[node] <= self.truck_capacity:
                current_trip.append(node)
                current_load += demands[node]
            else:
                current_trip.append(0)  # return to depot
                trips.append(current_trip)
                current_trip = [0, node]  # start new trip from depot with current node
                current_load = demands[node]
        current_trip.append(0)
        trips.append(current_trip)
        return trips

    def plot_truck_trips(self, trips, nodes):
        """
        Plot all truck trips on a 2D plot with unique colors and a legend.

        :param trips: List of trips; each trip is a list of node indices.
        :param nodes: 2D numpy array of node coordinates (first node is depot, then cluster centroids).
        """
        plt.figure(figsize=(8, 8))

        # Plot depot (node 0)
        plt.scatter(nodes[0, 0], nodes[0, 1], c="red", marker="s", s=100, label="Depot")

        # Plot cluster centroids (nodes 1..n)
        plt.scatter(nodes[1:, 0], nodes[1:, 1], c="blue", label="Cluster Centroids")

        # Create a color cycle for the trips
        cmap = get_cmap("tab10")  # or 'rainbow', etc.

        for idx, trip in enumerate(trips):
            # We'll skip the first node in the trip if it's just 0, but let's keep the entire route
            trip_coords = nodes[trip]
            color = cmap(idx % 10)  # pick color from colormap
            plt.plot(
                trip_coords[:, 0],
                trip_coords[:, 1],
                marker="o",
                color=color,
                label=f"Trip {idx+1}",
            )

        plt.title("Truck Trips (Multi-Trip Routing)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)

        # Create a combined legend. We use unique labels so we don't spam the legend with repeated "Depot" entries, etc.
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_handles_labels = dict(zip(labels, handles))
        plt.legend(
            unique_handles_labels.values(), unique_handles_labels.keys(), loc="best"
        )

        plt.show()


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())

    # Usage Example for TruckRouter
    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200  # Truck capacity (orders from depot to cluster must be within this capacity)
    
    router = TruckRouter(file_path, min_cluster_size, truck_capacity)
    
    # Load customer data, perform HDBSCAN clustering, and apply capacity-aware splitting
    depot, final_labels, final_centroids, cluster_demands = router.load_cluster_data()
    print("Depot Coordinates:")
    print(depot)
    print("Final Cluster Centroids:")
    print(final_centroids)
    print("Cluster Demands after Splitting:")
    print(cluster_demands)
    
    # Create nodes for truck routing: depot is first, then cluster centroids
    nodes = np.vstack([depot, final_centroids])
    
    # Create demands list: depot demand = 0, then demands in the order of sorted new cluster labels
    unique_labels = sorted(set(final_labels))
    demands = [0]
    for label in unique_labels:
        demands.append(cluster_demands[label])
    print("Demands for VRP (Depot then clusters):", demands)
    
    # Compute distance matrix among nodes (depot + centroids)
    distance_matrix = router.compute_distance_matrix(nodes)
    
    # Solve TSP on these nodes (using depot as starting and ending point)
    tsp_route = router.solve_tsp(distance_matrix)
    if tsp_route is None:
        print("TSP solver failed to find a solution.")
        sys.exit(1)
    print("TSP Route (node indices):", tsp_route)
    
    # Split the TSP route into multiple truck trips based on truck capacity
    trips = router.split_tsp_route_into_trips(tsp_route, demands)
    print("Truck Trips (each trip is a list of node indices):")
    for trip in trips:
        print(trip)
    
    # Plot the truck trips on the 2D map
    router.plot_truck_trips(trips, nodes)
