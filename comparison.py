"""
Comparison Module
-------------------
This module compares the baseline truck-only routing solution with an advanced truck routing solution 
that uses an ALNS heuristic over cluster centroids determined via HDBSCAN from the Solomon dataset.

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

# Import baseline VRP solution (truck-only) from utils
from baseline.truck_delivery_baseline import TruckDeliveryBaseline
from entities.truck import Truck

# Import CustomerAnalyzer from our customer_analysis module
from components.customer_clustering.customer_analysis import CustomerAnalyzer

# Import AdvancedTruckRouter from our advanced routing module
from components.truck_router.truck_router import AdvancedTruckRouter


def compute_baseline_metrics(file_path, num_trucks):
    """
    Compute metrics using the baseline truck-only routing solution.

    :param file_path: Path to the Solomon dataset.
    :param num_trucks: Number of trucks for the baseline solver.
    :return: Dictionary with baseline metrics and route details.
    """
    from utils.distance_matrix_calculator import DistanceMatrixCalculator

    # Load dataset and compute distance matrix
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

    # Calculate total distance, time, cost, emissions from baseline solution
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
    file_path, truck_speed=60, cost_per_km=0.5, cost_per_hour=20, emissions_per_km=120
):
    """
    Compute metrics using the advanced truck routing solution (ALNS on cluster centroids).
    This function loads the customer data, applies HDBSCAN to determine clusters,
    computes centroids, and then uses the AdvancedTruckRouter to compute an optimized route.

    :param file_path: Path to the Solomon dataset.
    :param truck_speed: Truck speed in km/h.
    :param cost_per_km: Cost per km.
    :param cost_per_hour: Cost per hour.
    :param emissions_per_km: Emissions per km (grams).
    :return: Dictionary with advanced router metrics and route details.
    """
    # Instantiate CustomerAnalyzer and load data
    analyzer = CustomerAnalyzer(file_path)
    customers_df = analyzer.load_data()

    # Use HDBSCAN for clustering
    labels, clusterer, n_clusters = analyzer.cluster_customers_hdbscan(
        min_cluster_size=5
    )

    # Compute centroids for each cluster (excluding depot)
    df_customers = customers_df[customers_df["ID"] != 0].copy()
    df_customers["Cluster"] = labels
    centroids = []
    for cluster_label, group in df_customers.groupby("Cluster"):
        if cluster_label == -1:
            # For noise points, add each point individually
            for _, row in group.iterrows():
                centroids.append([row["X"], row["Y"]])
        else:
            centroid = group[["X", "Y"]].mean().tolist()
            centroids.append(centroid)

    # Prepend and append the depot coordinates (assume depot has ID == 0)
    depot = customers_df[customers_df["ID"] == 0][["X", "Y"]].iloc[0].tolist()
    centroids = [depot] + centroids + [depot]

    # Use AdvancedTruckRouter on the computed centroids
    router = AdvancedTruckRouter(
        coords=centroids, iterations=1000, destruction_rate=0.3, random_seed=42
    )
    best_route, best_distance = router.solve_tsp_alns()

    # Calculate metrics using truck parameters
    total_distance = best_distance
    total_time = total_distance / truck_speed
    total_cost = total_distance * cost_per_km + total_time * cost_per_hour
    total_emissions = total_distance * emissions_per_km

    metrics = {
        "route": best_route,
        "total_distance": total_distance,
        "total_time": total_time,
        "total_cost": total_cost,
        "total_emissions": total_emissions,
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
        f"Total CO₂ Emissions (grams)  | {baseline_metrics['total_emissions']:.2f}           | {advanced_metrics['total_emissions']:.2f}\n"
    )


if __name__ == "__main__":
    # Path to the Solomon dataset file
    file_path = "dataset/c101.txt"

    # Set number of trucks for baseline (as in your baseline solution)
    num_trucks = 25  # From baseline output; note that only non-empty routes are considered later.

    # Compute baseline metrics using the baseline truck_delivery solution
    baseline_metrics = compute_baseline_metrics(file_path, num_trucks)
    print("\n--- Baseline Truck-Only Routing Metrics ---")
    print(f"Total Distance: {baseline_metrics['total_distance']:.2f} km")
    print(f"Total Time: {baseline_metrics['total_time']:.2f} hours")
    print(f"Total Cost: {baseline_metrics['total_cost']:.2f}")
    print(f"Total CO₂ Emissions: {baseline_metrics['total_emissions']:.2f} grams")

    # Compute advanced truck routing metrics using our ALNS solution on cluster centroids
    advanced_metrics = compute_advanced_router_metrics(file_path)
    print("\n--- Advanced Truck Routing Metrics (Truck as Mobile Depot) ---")
    print(f"Total Distance: {advanced_metrics['total_distance']:.2f} km")
    print(f"Total Time: {advanced_metrics['total_time']:.2f} hours")
    print(f"Total Cost: {advanced_metrics['total_cost']:.2f}")
    print(f"Total CO₂ Emissions: {advanced_metrics['total_emissions']:.2f} grams")

    # Print side-by-side comparison report
    print_comparison(baseline_metrics, advanced_metrics)

    # Optionally, visualize the advanced truck route
    from components.truck_router.truck_router import AdvancedTruckRouter

    router = AdvancedTruckRouter(
        coords=np.array(advanced_metrics["route"]), iterations=1000
    )  # This is for visualization only.
    # Note: Visualization should use the same centroids; here, for brevity, we skip replotting.
