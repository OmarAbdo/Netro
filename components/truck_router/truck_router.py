"""
TruckRouter Module (CVRP Version)
-----------------------------------
This module integrates customer analysis and truck routing to plan routes for trucks acting as mobile depots.
It performs the following steps:
    1. Loads customer data and clusters them using HDBSCAN via CustomerAnalyzer.
    2. Applies capacity-aware splitting (using CapacitatedClusterSplitter) so that each subcluster's total demand is ≤ truck capacity.
    3. Computes the subcluster centroids and aggregate demands.
    4. Constructs a VRP data model (with depot as node 0 and subsequent nodes as subcluster centroids).
    5. Solves the CVRP directly using OR-Tools, enforcing capacity constraints.
    6. Extracts and visualizes the resulting multi-vehicle routes.
    
Inspiration and References:
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For classical CVRP formulations; see p. 15, lines 5–10.)
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm", JIEM, p. 377, lines 10–15.
    - Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E, p. 67, lines 15–20.
      
This module is implemented in an OOP style with one class per file.
"""

import sys
import os
import time

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from components.customer_clustering.customer_analysis import CustomerAnalyzer
from components.customer_clustering.capacitated_clustering import (
    CapacitatedClusterSplitter,
)


class TruckRouter:
    """
    TruckRouter plans truck routes as a CVRP over the depot and the capacity-aware subcluster centroids.
    The truck acts as a mobile depot that, using multiple vehicles (trips), must cover the clusters subject to capacity constraints.
    """

    def __init__(self, file_path, min_cluster_size, truck_capacity, num_vehicles=None):
        """
        Initialize TruckRouter.

        :param file_path: Path to the Solomon dataset.
        :param min_cluster_size: Minimum cluster size for HDBSCAN.
        :param truck_capacity: Capacity of the truck.
        :param num_vehicles: Number of vehicles (trips) available. If None, set to a high number.
        """
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity
        # If num_vehicles is not specified, allow many vehicles so the solver can choose routes for only non-empty ones.
        self.num_vehicles = num_vehicles if num_vehicles is not None else 50

    def load_cluster_data(self):
        """
        Loads customer data, performs HDBSCAN clustering, and applies capacity-aware splitting.

        :return: Tuple (depot, final_labels, final_centroids, cluster_demands)
                 - depot: [X, Y] coordinates of the depot.
                 - final_labels: New cluster labels after splitting.
                 - final_centroids: Array of subcluster centroids.
                 - cluster_demands: Dictionary mapping new cluster label to its total demand.
        """
        analyzer = CustomerAnalyzer(self.file_path)
        customers_df = analyzer.load_data()

        # Depot assumed to be the first row (ID == 0)
        depot_row = customers_df[customers_df["ID"] == 0].iloc[0]
        depot = np.array([depot_row["X"], depot_row["Y"]])

        # Cluster customers (excluding depot) using HDBSCAN
        df_customers = customers_df[customers_df["ID"] != 0].copy()
        coords = df_customers[["X", "Y"]].values
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        initial_labels = clusterer.fit_predict(coords)
        print(
            f"HDBSCAN initially found {len(set(initial_labels)) - (1 if -1 in initial_labels else 0)} clusters (noise labeled as -1)."
        )

        # Apply capacity-aware splitting via CapacitatedClusterSplitter
        splitter = CapacitatedClusterSplitter(
            df_customers, initial_labels, self.truck_capacity
        )
        final_labels, final_centroids, subcluster_mapping, new_cluster_demands = (
            splitter.split_clusters()
        )
        print("Cluster Demands after Splitting:")
        print(new_cluster_demands)
        return depot, final_labels, final_centroids, new_cluster_demands

    def create_vrp_data_model(self, depot, centroids, cluster_demands):
        """
        Create the VRP data model using the depot and the capacity-aware subcluster centroids.
        Node 0 is the depot; nodes 1..n are the subcluster centroids.

        :param depot: [X, Y] coordinates of the depot.
        :param centroids: Array of subcluster centroids.
        :param cluster_demands: Dictionary mapping new cluster label to demand.
        :return: Data dictionary for the VRP.
        """
        # We need to ensure that the demands are ordered consistently with centroids.
        # Assume centroids are in the same order as sorted unique new cluster labels.
        unique_labels = sorted(cluster_demands.keys())
        demands = [0]  # Depot demand = 0
        for label in unique_labels:
            demands.append(cluster_demands[label])

        # Create nodes: first node is depot, then the centroids (in the order corresponding to unique_labels)
        nodes = np.vstack([depot, centroids])

        # Build full distance matrix (Euclidean)
        num_nodes = len(nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance_matrix[i, j] = np.linalg.norm(nodes[i] - nodes[j])

        # Data model for CVRP
        data = {
            "distance_matrix": distance_matrix.astype(
                int
            ).tolist(),  # convert to int for OR-Tools
            "demands": demands,
            "vehicle_capacities": [self.truck_capacity] * self.num_vehicles,
            "num_vehicles": self.num_vehicles,
            "depot": 0,
        }
        return data, nodes

    def solve_cvrp(self, data):
        """
        Solve the CVRP for the given data model using OR-Tools.

        :param data: Dictionary containing VRP data.
        :return: Dictionary with keys:
                 - 'routes': List of routes (each a list of node indices).
                 - 'total_distance': Total distance of all routes.
        """
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # Create distance callback.
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraint.
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # no slack
            data["vehicle_capacities"],
            True,  # start cumul at zero
            "Capacity",
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30

        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            return {"routes": None, "total_distance": None}

        # Extract solution.
        routes = []
        total_distance = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            route.append(manager.IndexToNode(index))
            if len(route) > 2:  # non-empty route (besides depot)
                routes.append(route)
                total_distance += route_distance
        return {"routes": routes, "total_distance": total_distance}

    def plot_truck_routes(self, routes, nodes):
        """
        Plot the CVRP solution routes on a 2D map.

        :param routes: List of routes (each a list of node indices).
        :param nodes: 2D numpy array of node coordinates (depot first, then centroids).
        """
        plt.figure(figsize=(8, 8))
        # Plot depot and clusters.
        plt.scatter(nodes[0, 0], nodes[0, 1], c="red", marker="s", s=100, label="Depot")
        plt.scatter(nodes[1:, 0], nodes[1:, 1], c="blue", label="Clusters")
        # Plot each route with a unique color.
        from matplotlib.cm import get_cmap

        cmap = get_cmap("tab10")
        for idx, route in enumerate(routes):
            color = cmap(idx % 10)
            route_coords = nodes[route]
            plt.plot(
                route_coords[:, 0],
                route_coords[:, 1],
                marker="o",
                color=color,
                label=f"Route {idx+1}",
            )
        plt.title("Truck Routes (CVRP Solution)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def solve(self):
        """
        Full workflow: load cluster data, build VRP data model, solve CVRP, and plot routes.

        :return: Dictionary with CVRP solution metrics.
        """
        depot, final_labels, final_centroids, cluster_demands = self.load_cluster_data()
        data, nodes = self.create_vrp_data_model(
            depot, final_centroids, cluster_demands
        )
        solution = self.solve_cvrp(data)
        if solution["routes"] is None:
            print("CVRP solver failed to find a solution.")
            return None
        # self.plot_truck_routes(solution["routes"], nodes)
        return solution


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())

    # Usage Example for advanced TruckRouter using CVRP formulation
    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200
    # Optionally, specify the maximum number of vehicles (trips)
    num_vehicles = 25

    router = TruckRouter(file_path, min_cluster_size, truck_capacity, num_vehicles)

    start_time = time.time()
    solution = router.solve()
    end_time = time.time()

    if solution is not None:
        print("CVRP Routes:")
        for idx, route in enumerate(solution["routes"]):
            print(f"Route {idx+1}: {route}")
        print(f"Total Distance: {solution['total_distance']}")
        print(f"Solver Runtime: {end_time - start_time:.2f} seconds")
