# netro/services/routing/netro_routing_service.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot
from netro.core.entities.cluster import Cluster
from netro.core.interfaces.routing import RoutingAlgorithm


class NetroRoutingService:
    """
    High-level service for the complete Netro routing solution.
    Orchestrates the truck routing to clusters and robot delivery within clusters.

    This service implements the Netro formula:
    T(Netro) = K * (t(travel-cluster) + t(unloading)) + [M/R * (t(travel-robot-customer) + t(robot-service)) + t(recovery)]

    where:
    - K: number of clusters
    - M: customers per cluster
    - R: robots per cluster
    """

    def __init__(
        self,
        truck_routing_algorithm: RoutingAlgorithm,
        robot_routing_service: Any,  # Using Any to avoid circular import
        robot_unloading_time: float = 5.0,
    ):
        """
        Initialize the Netro routing service.

        Args:
            truck_routing_algorithm: Algorithm for routing trucks between clusters.
            robot_routing_service: Service for routing robots within clusters.
            robot_unloading_time: Time to unload all robots from a truck in minutes.
        """
        self.truck_routing_algorithm = truck_routing_algorithm
        self.robot_routing_service = robot_routing_service
        self.robot_unloading_time = robot_unloading_time

    def solve(
        self,
        depot: Location,
        clusters: List[Cluster],
        centroids: Dict[int, Location],
        trucks: List[Truck],
        robots_per_truck: List[List[Robot]],
    ) -> Dict[str, Any]:
        """
        Solve the complete Netro routing problem.

        Args:
            depot: Depot location.
            clusters: List of customer clusters.
            centroids: Dictionary mapping cluster ID to centroid location.
            trucks: List of available trucks.
            robots_per_truck: List of robots available for each truck.

        Returns:
            Dictionary with solution details and metrics.
        """
        # First, route trucks between cluster centroids
        truck_routes, truck_metrics = self._route_trucks(depot, centroids, trucks)

        # Then, for each truck route, route robots within each cluster
        cluster_routes = {}
        cluster_metrics = {}

        for truck_idx, truck_route in enumerate(truck_routes):
            truck_route_metrics = {
                "cluster_times": [],
                "total_robot_distance": 0.0,
                "total_time": 0.0,
            }

            truck = trucks[truck_idx]
            truck_robots = robots_per_truck[truck_idx]

            for location_idx in truck_route:
                # Skip depot
                if location_idx == 0:
                    continue

                # Get the cluster ID from the centroid location ID (-id-1000)
                cluster_id = -centroids[location_idx].id - 1000

                if cluster_id not in clusters:
                    continue

                cluster = clusters[cluster_id]
                centroid = centroids[location_idx]

                # Route robots within this cluster
                robot_routes, robot_metrics = self.robot_routing_service.route_robots(
                    cluster_centroid=centroid,
                    customers=cluster.locations,
                    robots=truck_robots,
                    truck=truck,
                )

                # Store the robot routes for this cluster
                cluster_routes[cluster_id] = robot_routes

                # Update metrics
                cluster_time = (
                    self.robot_unloading_time + robot_metrics["max_robot_time"]
                )
                truck_route_metrics["cluster_times"].append(cluster_time)
                truck_route_metrics["total_robot_distance"] += robot_metrics[
                    "total_robot_distance"
                ]
                truck_route_metrics["total_time"] += cluster_time

            # Store metrics for this truck route
            cluster_metrics[truck_idx] = truck_route_metrics

        # Calculate overall metrics
        total_time = truck_metrics["total_time"]
        for truck_idx, metrics in cluster_metrics.items():
            total_time += metrics["total_time"]

        # Create and return the complete solution
        solution = {
            "truck_routes": truck_routes,
            "cluster_routes": cluster_routes,
            "truck_metrics": truck_metrics,
            "cluster_metrics": cluster_metrics,
            "total_time": total_time,
            "total_truck_distance": truck_metrics["total_distance"],
            "total_robot_distance": sum(
                m["total_robot_distance"] for m in cluster_metrics.values()
            ),
        }

        return solution

    def _route_trucks(
        self, depot: Location, centroids: Dict[int, Location], trucks: List[Truck]
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Route trucks between cluster centroids.

        Args:
            depot: Depot location.
            centroids: Dictionary mapping cluster ID to centroid location.
            trucks: List of available trucks.

        Returns:
            A tuple containing:
            - List of truck routes.
            - Dictionary with metrics.
        """
        # Create list of locations with depot as first location
        centroid_locations = list(centroids.values())
        all_locations = [depot] + centroid_locations

        # Compute distance matrix
        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations))

        for i in range(n_locations):
            for j in range(n_locations):
                distance_matrix[i, j] = all_locations[i].distance_to(all_locations[j])

        # Extract demands
        demands = np.array([loc.demand for loc in all_locations])

        # Truck capacities
        capacities = np.array([truck.capacity for truck in trucks])

        # Solve the routing problem
        routes, total_distance = self.truck_routing_algorithm.solve(
            distance_matrix=distance_matrix,
            demands=demands,
            capacities=capacities,
            depot_index=0,
            max_vehicles=len(trucks),
        )

        # Calculate truck metrics
        total_time = 0.0

        for i, route in enumerate(routes):
            if not route:
                continue

            # Get the truck for this route
            truck = trucks[min(i, len(trucks) - 1)]

            # Calculate route distance
            route_distance = 0.0
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]

            # Calculate route time
            route_time = route_distance / truck.speed
            total_time += route_time

        metrics = {"total_distance": total_distance, "total_time": total_time}

        return routes, metrics
