"""
Comparison Module
-------------------
This module compares the baseline truck-only routing solution with an advanced truck routing solution 
that uses a clustering and multi-trip approach (with capacity-aware splitting) over cluster centroids 
determined via HDBSCAN from the Solomon dataset.

Inspiration and References:
    - Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems", 
      Computers and Electrical Engineering, 67, 596–607. (For ALNS framework; see p. 600, lines 5–15.)
    - Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm",
      JIEM, p. 377, lines 10–15. (For clustering-based fleet sizing ideas.)
    - Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
      (For baseline VRP formulations.)
      
This module outputs metrics (total distance, travel time, cost, CO₂ emissions) for both approaches for direct comparison.
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Import baseline VRP solution (truck-only) from utils
from baseline.truck_delivery_baseline import TruckDeliveryBaseline
from entities.truck import Truck
from utils.distance_matrix_calculator import DistanceMatrixCalculator

# Import CustomerAnalyzer from our customer_clustering module (which now uses HDBSCAN)
from components.customer_clustering.customer_analysis import CustomerAnalyzer

# Import the updated TruckRouter from our truck_router module (advanced routing solution)
from components.truck_router.truck_router import TruckRouter


def compute_baseline_metrics(file_path, num_trucks):
    """
    Compute metrics using the baseline truck-only routing solution.

    :param file_path: Path to the Solomon dataset.
    :param num_trucks: Number of trucks for the baseline solver.
    :return: Dictionary with baseline metrics and route details.
    """
    calculator = DistanceMatrixCalculator(file_path)
    calculator.load_data()
    distance_matrix = calculator.compute_distance_matrix()
    demands = calculator.get_demands()
    ready_times = calculator.get_ready_times()  # Unused in baseline
    due_times = calculator.get_due_times()  # Unused in baseline
    service_times = calculator.get_service_times()  # Unused in baseline

    trucks = [Truck(truck_id=i) for i in range(num_trucks)]

    # Solve baseline VRP using OR-Tools-based approach
    baseline_solver = TruckDeliveryBaseline(
        distance_matrix, trucks, demands, ready_times, due_times, service_times
    )
    result = baseline_solver.solve_vrp()
    if result["routes"] is None:
        raise Exception("Baseline solver did not find a solution.")

    # Calculate metrics
    total_distance = result["total_distance"]
    total_time = 0.0
    total_cost = 0.0
    total_emissions = 0.0
    for idx, route in enumerate(result["routes"]):
        distance = 0.0
        for j in range(len(route) - 1):
            distance += distance_matrix[route[j]][route[j + 1]]
        truck = trucks[idx]
        time_needed = distance / truck.speed
        cost = distance * truck.cost_per_km + time_needed * truck.cost_per_hour
        emissions = distance * truck.emissions_per_km

        total_time += time_needed
        total_cost += cost
        total_emissions += emissions

    metrics = {
        "routes": result["routes"],
        "total_distance": total_distance,
        "total_time": total_time,
        "total_cost": total_cost,
        "total_emissions": total_emissions,
    }
    return metrics


def compute_advanced_router_metrics(
    file_path,
    min_cluster_size=5,
    truck_capacity=200,
    truck_speed=60,
    cost_per_km=0.5,
    cost_per_hour=20,
    emissions_per_km=120,
):
    """
    Compute metrics using the advanced truck routing solution.
    This function:
        1. Loads customer data via CustomerAnalyzer.
        2. Uses HDBSCAN to cluster customers and then applies capacity-aware splitting.
        3. Computes cluster centroids and aggregate demands.
        4. Solves a TSP (via OR-Tools) on the depot and the cluster centroids.
        5. Splits the TSP route into multiple truck trips based on truck capacity.
    The truck functions as a mobile depot for launching robots.

    :param file_path: Path to the Solomon dataset.
    :param min_cluster_size: Minimum cluster size for HDBSCAN.
    :param truck_capacity: Truck capacity.
    :param truck_speed: Truck speed in km/h.
    :param cost_per_km: Cost per km.
    :param cost_per_hour: Cost per hour.
    :param emissions_per_km: Emissions per km (grams).
    :return: Dictionary with advanced router metrics and route details.
    """
    start_time = time.time()

    router = TruckRouter(file_path, min_cluster_size, truck_capacity)
    depot, final_labels, final_centroids, cluster_demands = router.load_cluster_data()
    print("Cluster Demands after Splitting (Advanced):", cluster_demands)

    # Build nodes array: depot as first node, then subcluster centroids
    nodes = np.vstack([depot, final_centroids])

    # Build demands array: depot = 0, then for each subcluster label in sorted order
    unique_labels = sorted(set(final_labels))
    demands = [0]
    for label in unique_labels:
        demands.append(cluster_demands[label])
    print("Demands for VRP (Depot then clusters):", demands)

    # Compute distance matrix among nodes (depot + centroids)
    distance_matrix = router.compute_distance_matrix(nodes)

    # Solve TSP on these nodes (depot as start and end)
    tsp_route = router.solve_tsp(distance_matrix)
    if tsp_route is None:
        return {
            "time_sec": time.time() - start_time,
            "total_distance": math.inf,
            "status": "NoSolution",
        }
    print("TSP Route (node indices):", tsp_route)

    # Split TSP route into truck trips based on truck capacity
    trips = router.split_tsp_route_into_trips(tsp_route, demands)
    print("Truck Trips (each trip is a list of node indices):")
    for trip in trips:
        print(trip)

    # For consistency, total_distance is taken from the TSP route (you might also compute actual multi-trip distance)
    total_distance = 0.0
    for i in range(len(tsp_route) - 1):
        total_distance += distance_matrix[tsp_route[i]][tsp_route[i + 1]]

    end_time = time.time()
    total_time = total_distance / truck_speed
    total_cost = total_distance * cost_per_km + total_time * cost_per_hour
    total_emissions = total_distance * emissions_per_km

    metrics = {
        "route": tsp_route,
        "trips": trips,
        "total_distance": total_distance,
        "total_time": total_time,
        "total_cost": total_cost,
        "total_emissions": total_emissions,
        "runtime_sec": end_time - start_time,
        "status": "OK",
    }
    return metrics


def print_comparison(baseline_metrics, advanced_metrics):
    """
    Print a side-by-side comparison of the baseline and advanced truck routing metrics.

    :param baseline_metrics: Dictionary with metrics from the baseline solver.
    :param advanced_metrics: Dictionary with metrics from the advanced truck routing solver.
    """
    print("\n--- Comparison Report ---\n")
    print(
        "Metric                       | Baseline (Truck-Only) | Advanced (Truck + Clusters)"
    )
    print(
        "-----------------------------|-----------------------|-----------------------------"
    )
    print(
        f"Total Distance (km)          | {baseline_metrics['total_distance']:.2f}                | {advanced_metrics['total_distance']:.2f}"
    )
    print(
        f"Total Time (hours)           | {baseline_metrics['total_time']:.2f}                | {advanced_metrics['total_time']:.2f}"
    )
    print(
        f"Total Cost                   | {baseline_metrics['total_cost']:.2f}                | {advanced_metrics['total_cost']:.2f}"
    )
    print(
        f"Total CO₂ Emissions (grams)  | {baseline_metrics['total_emissions']:.2f}           | {advanced_metrics['total_emissions']:.2f}"
    )
    print(
        f"Runtime (sec)                | N/A                   | {advanced_metrics['runtime_sec']:.2f}\n"
    )


if __name__ == "__main__":
    file_path = "dataset/c101.txt"

    # Compute baseline metrics using the baseline truck-only approach
    num_trucks = 25  # as used in baseline
    baseline_metrics = compute_baseline_metrics(file_path, num_trucks)
    print("\n--- Baseline Truck-Only Routing Metrics ---")
    print(f"Total Distance: {baseline_metrics['total_distance']:.2f} km")
    print(f"Total Time: {baseline_metrics['total_time']:.2f} hours")
    print(f"Total Cost: {baseline_metrics['total_cost']:.2f}")
    print(f"Total CO₂ Emissions: {baseline_metrics['total_emissions']:.2f} grams")

    # Compute advanced metrics using the advanced truck routing solution
    advanced_metrics = compute_advanced_router_metrics(file_path)
    print("\n--- Advanced Truck Routing Metrics (Truck as Mobile Depot) ---")
    print(f"Total Distance: {advanced_metrics['total_distance']:.2f} km")
    print(f"Total Time: {advanced_metrics['total_time']:.2f} hours")
    print(f"Total Cost: {advanced_metrics['total_cost']:.2f}")
    print(f"Total CO₂ Emissions: {advanced_metrics['total_emissions']:.2f} grams")
    print(f"Runtime: {advanced_metrics['runtime_sec']:.2f} seconds")

    # Print side-by-side comparison
    print_comparison(baseline_metrics, advanced_metrics)
