"""
TruckRouter Module (CVRP Version)
----------------------------------
This module integrates customer analysis and truck routing to plan routes for trucks acting as mobile depots.
It performs the following steps:
    1. Loads customer data and clusters them using the updated CustomerAnalyzer (with noise reassignment).
    2. Applies capacity-aware splitting (using CapacitatedClusterSplitter) so that each subcluster's total demand is ≤ truck capacity.
    3. Computes the subcluster centroids and aggregate demands.
    4. Constructs a VRP data model (with depot as node 0 and subsequent nodes as subcluster centroids, with their demands).
    5. Solves the CVRP directly using OR‑Tools (without additional local search enhancements).
    6. Returns the solution so that plotting can be performed separately.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys
import os

sys.path.append(os.getcwd())
from netro.components.customer_clustering.customer_analysis import CustomerAnalyzer
from netro.components.customer_clustering.capacitated_clustering import (
    CapacitatedClusterSplitter,
)


class TruckRouter:
    def __init__(self, file_path, min_cluster_size, truck_capacity, num_vehicles=None):
        self.file_path = file_path
        self.min_cluster_size = min_cluster_size
        self.truck_capacity = truck_capacity
        self.num_vehicles = num_vehicles if num_vehicles is not None else 50

    def load_cluster_data(self):
        analyzer = CustomerAnalyzer(self.file_path)
        customers_df = analyzer.load_data()
        depot_row = customers_df[customers_df["ID"] == 0].iloc[0]
        depot = np.array([depot_row["X"], depot_row["Y"]])
        df_customers = customers_df[customers_df["ID"] != 0].copy()
        labels, clusterer, n_clusters = analyzer.cluster_customers_hdbscan(
            min_cluster_size=self.min_cluster_size
        )
        print(f"Clusters after noise reassignment: {n_clusters}")
        splitter = CapacitatedClusterSplitter(df_customers, labels, self.truck_capacity)
        final_labels, final_centroids, subcluster_mapping, new_cluster_demands = (
            splitter.split_clusters()
        )
        print("Cluster Demands after Splitting:")
        print(new_cluster_demands)
        return depot, final_labels, final_centroids, new_cluster_demands

    def create_vrp_data_model(self, depot, centroids, cluster_demands):
        unique_labels = sorted(cluster_demands.keys())
        demands = [0]  # Depot demand is 0
        for label in unique_labels:
            demands.append(cluster_demands[label])
        nodes = np.vstack([depot, centroids])
        num_nodes = len(nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance_matrix[i, j] = np.linalg.norm(nodes[i] - nodes[j])
        data = {
            "distance_matrix": distance_matrix.astype(int).tolist(),
            "demands": demands,
            "vehicle_capacities": [self.truck_capacity] * self.num_vehicles,
            "num_vehicles": self.num_vehicles,
            "depot": 0,
        }
        return data, nodes

    def solve_cvrp(self, data):
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_idx, to_idx: data["distance_matrix"][
                manager.IndexToNode(from_idx)
            ][manager.IndexToNode(to_idx)]
        )
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        demand_callback_index = routing.RegisterUnaryTransitCallback(
            lambda from_idx: data["demands"][manager.IndexToNode(from_idx)]
        )
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, data["vehicle_capacities"], True, "Capacity"
        )
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.seconds = 60
        solution = routing.SolveWithParameters(search_parameters)
        if not solution:
            return {"routes": None, "total_distance": None}
        routes = []
        total_distance = 0
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            route.append(manager.IndexToNode(index))
            if len(route) > 2:
                routes.append(route)
        return {"routes": routes, "total_distance": total_distance}

    def plot_truck_routes(self, routes, nodes):
        plt.figure(figsize=(8, 8))
        plt.scatter(nodes[0, 0], nodes[0, 1], c="red", marker="s", s=100, label="Depot")
        plt.scatter(nodes[1:, 0], nodes[1:, 1], c="blue", label="Clusters")
        cmap = plt.cm.get_cmap("tab10")
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
        depot, final_labels, final_centroids, cluster_demands = self.load_cluster_data()
        data, nodes = self.create_vrp_data_model(
            depot, final_centroids, cluster_demands
        )
        solution = self.solve_cvrp(data)
        if solution["routes"] is None:
            print("CVRP solver failed to find a solution.")
            return None, nodes
        return solution, nodes


if __name__ == "__main__":
    file_path = "dataset/c101.txt"
    min_cluster_size = 5
    truck_capacity = 200
    num_vehicles = 25
    router = TruckRouter(file_path, min_cluster_size, truck_capacity, num_vehicles)
    start_time = time.time()
    solution, nodes = router.solve()
    end_time = time.time()
    if solution is not None:
        print("CVRP Routes:")
        for idx, route in enumerate(solution["routes"]):
            print(f"Route {idx+1}: {route}")
        print(f"Total Distance: {solution['total_distance']}")
        print(f"Solver Runtime: {end_time - start_time:.2f} seconds")
        router.plot_truck_routes(solution["routes"], nodes)
