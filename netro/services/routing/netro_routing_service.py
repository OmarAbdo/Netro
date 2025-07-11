# netro/services/routing/netro_routing_service.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot
from netro.core.entities.cluster import Cluster
from netro.core.interfaces.routing import RoutingAlgorithm
from netro.services.routing.cluster_route_handler import ClusterRouteHandler
from netro.services.routing.truck_route_planner import TruckRoutePlanner


class NetroRoutingService:
    """
    High-level service for the complete Netro routing solution.
    Orchestrates the truck routing to clusters and robot delivery within clusters.

    This service implements the Netro formula:
    T(Netro) = max_over_trucks(t(truck_travel) + t(robot_operations_parallel))

    where:
    - Multiple trucks operate simultaneously on different routes
    - Within each cluster, robots operate in parallel
    - Driver waits for all robots to return before moving to next cluster
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
        self.truck_route_planner = TruckRoutePlanner(truck_routing_algorithm)
        self.cluster_route_handler = ClusterRouteHandler(
            robot_routing_service, robot_unloading_time
        )

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
        # First, create a mapping from centroid indices to cluster IDs
        centroid_idx_to_cluster_id, cluster_id_to_centroid_idx = (
            self._create_centroid_mappings(clusters, centroids)
        )

        # Route trucks between cluster centroids (with cluster validation)
        truck_routes, truck_metrics = self.truck_route_planner.plan_routes(
            depot, centroids, trucks
        )
        
        # Verify all clusters are assigned
        assigned_centroids = {idx for route in truck_routes for idx in route[1:-1]}
        all_centroid_indices = set(centroids.keys())
        unassigned = all_centroid_indices - assigned_centroids
        
        if unassigned:
            print(f"Found {len(unassigned)} unassigned clusters, adding forced routes")
            # Create emergency routes for remaining clusters
            for cluster_id in unassigned:
                truck_idx = len(truck_routes) % len(trucks)
                truck_routes.append([0, cluster_id, 0])
                truck_metrics["total_distance"] += centroids[cluster_id].distance_to(depot) * 2
                truck_metrics["total_time"] += (centroids[cluster_id].distance_to(depot) * 2) / trucks[truck_idx].speed

        # Process each truck route and handle robot routing within clusters
        cluster_results = self.cluster_route_handler.process_routes(
            truck_routes,
            trucks,
            robots_per_truck,
            clusters,
            centroids,
            centroid_idx_to_cluster_id,
        )

        # Check for unassigned clusters
        assigned_clusters = set(cluster_results["assigned_clusters"])
        unassigned_clusters = set(c.id for c in clusters) - assigned_clusters

        # Handle any unassigned clusters
        if unassigned_clusters:
            print(
                f"Found {len(unassigned_clusters)} unassigned clusters, creating additional routes"
            )
            additional_results = self.cluster_route_handler.handle_unassigned_clusters(
                unassigned_clusters,
                clusters,
                centroids,
                depot,
                trucks,
                robots_per_truck,
                cluster_results["cluster_routes"],
                cluster_results["cluster_metrics"],
                cluster_id_to_centroid_idx,
                len(truck_routes),
            )

            # Merge the additional results with existing ones
            if additional_results:
                truck_routes.extend(additional_results["truck_routes"])
                truck_metrics["total_distance"] += additional_results["total_distance"]
                truck_metrics["total_time"] += additional_results["total_time"]
                cluster_results["cluster_routes"].update(
                    additional_results["cluster_routes"]
                )
                cluster_results["cluster_metrics"].update(
                    additional_results["cluster_metrics"]
                )

        # Calculate metrics with proper parallel time consideration
        # FIXED: Calculate parallel time correctly - trucks operate simultaneously
        parallel_time = self._calculate_parallel_time(
            truck_routes, cluster_results["cluster_metrics"], trucks, depot, centroids
        )

        # Calculate sequential time for comparison
        sequential_time = truck_metrics["total_time"] + sum(
            m.get("total_time", 0) for m in cluster_results["cluster_metrics"].values()
        )

        # Create the complete solution
        solution = {
            "truck_routes": truck_routes,
            "cluster_routes": cluster_results["cluster_routes"],
            "truck_metrics": truck_metrics,
            "cluster_metrics": cluster_results["cluster_metrics"],
            "total_time": parallel_time,  # This is the realistic total time
            "parallel_time": parallel_time,
            "sequential_time": sequential_time,
            "total_truck_distance": truck_metrics["total_distance"],
            "total_robot_distance": sum(
                m.get("total_robot_distance", 0)
                for m in cluster_results["cluster_metrics"].values()
            ),
        }

        return solution

    def _calculate_parallel_time(
        self,
        truck_routes: List[List[int]],
        cluster_metrics: Dict[int, Dict[str, float]],
        trucks: List[Truck],
        depot: Location,
        centroids: Dict[int, Location],
    ) -> float:
        """
        Calculate the total time for parallel truck operations.

        CORRECTED: For each truck route, calculate truck_travel_time + robot_operation_time
        Then take the maximum across all routes (since trucks operate in parallel).

        Args:
            truck_routes: List of truck routes
            cluster_metrics: Metrics for cluster operations
            trucks: List of trucks
            depot: Depot location
            centroids: Dictionary of centroids

        Returns:
            Maximum time among all truck routes (parallel operation)
        """
        route_times = []

        for truck_idx, route in enumerate(truck_routes):
            if len(route) <= 2:  # Skip routes that only visit depot
                continue

            # Calculate truck travel time for this specific route
            truck = trucks[min(truck_idx, len(trucks) - 1)]
            truck_travel_time = self._calculate_actual_truck_route_time(
                route, truck, depot, centroids
            )

            # Get robot operation time for this truck's clusters
            # This should be the time spent at clusters, not additional travel
            cluster_operation_time = 0.0
            if truck_idx in cluster_metrics:
                metrics = cluster_metrics[truck_idx]
                # Use max_time (parallel robot operation within clusters)
                cluster_operation_time = metrics.get("max_time", 0.0)

            # CORRECTED: Total time for this truck = travel time + cluster operation time
            total_route_time = truck_travel_time + cluster_operation_time
            route_times.append(total_route_time)

            print(
                f"Truck {truck_idx}: Travel={truck_travel_time:.2f}h, Cluster={cluster_operation_time:.2f}h, Total={total_route_time:.2f}h"
            )

        # Since trucks operate in parallel, total time is the maximum route time
        max_time = max(route_times) if route_times else 0.0
        print(f"Parallel time calculation: max({route_times}) = {max_time:.2f}h")
        return max_time

    def _create_centroid_mappings(
        self, clusters: List[Cluster], centroids: Dict[int, Location]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Create reliable mappings between centroid indices and cluster IDs.

        Args:
            clusters: List of customer clusters.
            centroids: Dictionary mapping cluster ID to centroid location.

        Returns:
            Tuple of two dictionaries:
            - centroid_idx_to_cluster_id: Maps route indices to cluster IDs
            - cluster_id_to_centroid_idx: Maps cluster IDs to route indices
        """
        centroid_idx_to_cluster_id = {}
        cluster_id_to_centroid_idx = {}

        # Build mapping by matching coordinates
        for idx, (centroid_id, centroid) in enumerate(centroids.items(), start=1):
            for cluster in clusters:
                # Match by coordinates
                centroid_coords = (centroid.x, centroid.y)
                cluster_coords = (
                    cluster.centroid if cluster.centroid is not None else None
                )

                if cluster_coords is not None and np.allclose(
                    cluster_coords, centroid_coords, atol=1.0
                ):
                    centroid_idx_to_cluster_id[idx] = cluster.id
                    cluster_id_to_centroid_idx[cluster.id] = idx
                    break

                # Fallback to direct ID matching if coordinate matching fails
                if centroid_id == cluster.id:
                    centroid_idx_to_cluster_id[idx] = cluster.id
                    cluster_id_to_centroid_idx[cluster.id] = idx
                    break

        return centroid_idx_to_cluster_id, cluster_id_to_centroid_idx

    def _calculate_actual_truck_route_time(
        self,
        route: List[int],
        truck: Truck,
        depot: Location,
        centroids: Dict[int, Location],
    ) -> float:
        """
        Calculate time for a specific truck route by actually tracing the route.

        Args:
            route: List of location indices in the route
            truck: Truck assigned to this route
            depot: Depot location
            centroids: Dictionary mapping cluster ID to centroid location

        Returns:
            Total time for the truck route
        """
        if len(route) <= 2:  # Only depot
            return 0.0

        total_distance = 0.0
        current_location = depot

        # Calculate distance for each segment of the route
        for i in range(1, len(route) - 1):  # Skip depot at start and end
            route_idx = route[i]

            # Find the actual centroid for this route index
            # We need to map route indices back to actual centroids
            next_location = None

            # Try to find centroid by matching route index to centroid
            centroid_list = list(centroids.values())
            if route_idx - 1 < len(centroid_list):
                next_location = centroid_list[route_idx - 1]
            else:
                # Fallback: use the first available centroid
                next_location = next(iter(centroids.values())) if centroids else depot

            if next_location and next_location != current_location:
                segment_distance = current_location.distance_to(next_location)
                total_distance += segment_distance
                current_location = next_location

        # Return to depot
        if current_location != depot:
            total_distance += current_location.distance_to(depot)

        # Calculate time based on truck speed
        travel_time = total_distance / truck.speed
        return travel_time
