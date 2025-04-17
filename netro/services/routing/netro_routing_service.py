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

        # Route trucks between cluster centroids
        truck_routes, truck_metrics = self.truck_route_planner.plan_routes(
            depot, centroids, trucks
        )

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

        # Calculate metrics with parallel time consideration
        total_truck_time = truck_metrics["total_time"]

        # Calculate parallel time - this is a key fix
        # In each route, robots work in parallel, so we take the maximum robot time per route
        parallel_time = 0
        for truck_idx, metrics in cluster_results["cluster_metrics"].items():
            # For each truck route, add truck travel time + maximum cluster operation time
            route_idx = min(truck_idx, len(truck_routes) - 1) if truck_routes else 0
            if route_idx < len(truck_routes) and len(truck_routes[route_idx]) > 2:
                # Get truck time for this route
                truck_route_time = self._calculate_truck_route_time(
                    truck_routes[route_idx],
                    trucks[min(truck_idx, len(trucks) - 1)],
                    depot,
                    centroids,
                )
                # Add max cluster time for this route (parallel robot operation)
                parallel_time += truck_route_time + metrics.get("max_time", 0)

        # Create the complete solution
        solution = {
            "truck_routes": truck_routes,
            "cluster_routes": cluster_results["cluster_routes"],
            "truck_metrics": truck_metrics,
            "cluster_metrics": cluster_results["cluster_metrics"],
            "total_time": parallel_time,
            "parallel_time": parallel_time,
            "sequential_time": total_truck_time
            + sum(
                m.get("total_time", 0)
                for m in cluster_results["cluster_metrics"].values()
            ),
            "total_truck_distance": truck_metrics["total_distance"],
            "total_robot_distance": sum(
                m.get("total_robot_distance", 0)
                for m in cluster_results["cluster_metrics"].values()
            ),
        }

        return solution

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

    def _calculate_truck_route_time(
        self,
        route: List[int],
        truck: Truck,
        depot: Location,
        centroids: Dict[int, Location],
    ) -> float:
        """
        Calculate time for a truck route.

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

        # Create list of actual locations
        locations = [depot]
        for idx in route[1:-1]:  # Skip depot at start and end
            # Find the centroid for this route index
            # This is a simplification; in a real system we'd need a more robust lookup
            for centroid in centroids.values():
                locations.append(centroid)
                break
        locations.append(depot)  # Return to depot

        # Calculate total distance
        total_distance = 0.0
        for i in range(len(locations) - 1):
            total_distance += locations[i].distance_to(locations[i + 1])

        # Calculate time
        return total_distance / truck.speed
