# netro/services/routing/cluster_route_handler.py
from typing import List, Dict, Tuple, Any, Set
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot
from netro.core.entities.cluster import Cluster


class ClusterRouteHandler:
    """
    Handles routing of robots within clusters.

    Based on:
    Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system",
    Transportation Research Part E.
    """

    def __init__(self, robot_routing_service: Any, robot_unloading_time: float = 5.0):
        """
        Initialize the cluster route handler.

        Args:
            robot_routing_service: Service for routing robots within clusters.
            robot_unloading_time: Time to unload all robots from a truck in minutes.
        """
        self.robot_routing_service = robot_routing_service
        self.robot_unloading_time = robot_unloading_time

    def process_routes(
        self,
        truck_routes: List[List[int]],
        trucks: List[Truck],
        robots_per_truck: List[List[Robot]],
        clusters: List[Cluster],
        centroids: Dict[int, Location],
        centroid_idx_to_cluster_id: Dict[int, int],
    ) -> Dict[str, Any]:
        """
        Process each truck route and handle robot routing within clusters.

        Args:
            truck_routes: List of truck routes.
            trucks: List of available trucks.
            robots_per_truck: List of robots available for each truck.
            clusters: List of customer clusters.
            centroids: Dictionary mapping cluster ID to centroid location.
            centroid_idx_to_cluster_id: Mapping from route indices to cluster IDs.

        Returns:
            Dictionary with cluster routes, metrics, and list of assigned clusters.
        """
        # Build a list of all locations (depot + centroids)
        all_locations = [
            next(iter(centroids.values())).coordinates()
        ]  # Placeholder for depot
        for centroid in centroids.values():
            all_locations.append(centroid)

        # Initialize result dictionaries
        cluster_routes = {}
        cluster_metrics = {}
        assigned_clusters = set()

        # Process each truck route
        for truck_idx, truck_route in enumerate(truck_routes):
            if len(truck_route) <= 2:  # Skip routes that only visit depot
                continue

            # Initialize metrics for this truck route
            truck_route_metrics = {
                "cluster_times": [],
                "total_robot_distance": 0.0,
                "total_time": 0.0,
                "max_time": 0.0,  # Track maximum time for parallel operation
            }

            # Get the truck and robots for this route
            truck = trucks[min(truck_idx, len(trucks) - 1)]
            truck_robots = robots_per_truck[min(truck_idx, len(robots_per_truck) - 1)]

            # Process each stop in the route (skip depot at start and end)
            for location_idx in truck_route[1:-1]:
                # Find the cluster ID using our mapping
                if location_idx not in centroid_idx_to_cluster_id:
                    print(
                        f"Warning: No cluster found for location index {location_idx}, skipping"
                    )
                    continue

                cluster_id = centroid_idx_to_cluster_id[location_idx]
                assigned_clusters.add(cluster_id)

                # Find the cluster object
                cluster_obj = next((c for c in clusters if c.id == cluster_id), None)

                if cluster_obj is None:
                    print(
                        f"Warning: No cluster object found for ID {cluster_id}, skipping"
                    )
                    continue

                # Get the centroid for this cluster
                centroid = centroids.get(cluster_id)
                if not centroid:
                    # Try to find by location_idx if direct lookup fails
                    centroid = self._find_centroid_by_index(location_idx, centroids)
                    if not centroid:
                        print(
                            f"Warning: No centroid found for cluster {cluster_id}, skipping"
                        )
                        continue

                # Route robots within this cluster
                try:
                    robot_routes, robot_metrics = (
                        self.robot_routing_service.route_robots(
                            cluster_centroid=centroid,
                            customers=cluster_obj.locations,
                            robots=truck_robots,
                            truck=truck,
                        )
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
                    truck_route_metrics["max_time"] = max(
                        truck_route_metrics["max_time"], cluster_time
                    )

                except Exception as e:
                    print(f"Error routing robots in cluster {cluster_id}: {str(e)}")

            # Store metrics for this truck route
            cluster_metrics[truck_idx] = truck_route_metrics

        return {
            "cluster_routes": cluster_routes,
            "cluster_metrics": cluster_metrics,
            "assigned_clusters": list(assigned_clusters),
        }

    def handle_unassigned_clusters(
        self,
        unassigned_clusters: Set[int],
        clusters: List[Cluster],
        centroids: Dict[int, Location],
        depot: Location,
        trucks: List[Truck],
        robots_per_truck: List[List[Robot]],
        existing_cluster_routes: Dict[int, List[List[int]]],
        existing_cluster_metrics: Dict[int, Dict[str, float]],
        cluster_id_to_centroid_idx: Dict[int, int],
        next_route_idx: int,
    ) -> Dict[str, Any]:
        """
        Handle clusters that weren't assigned to any truck route.

        Args:
            unassigned_clusters: Set of cluster IDs that weren't assigned
            clusters: List of all clusters
            centroids: Dictionary mapping cluster ID to centroid location
            depot: Depot location
            trucks: List of available trucks
            robots_per_truck: List of robots available for each truck
            existing_cluster_routes: Existing cluster routes to update
            existing_cluster_metrics: Existing cluster metrics to update
            cluster_id_to_centroid_idx: Mapping from cluster IDs to centroid indices
            next_route_idx: Next available route index

        Returns:
            Dictionary with additional routes and metrics
        """
        print(f"Creating routes for {len(unassigned_clusters)} unassigned clusters")

        # Check if we have available trucks
        if not trucks:
            print("No available trucks for unassigned clusters")
            return None

        additional_truck_routes = []
        additional_cluster_routes = {}
        additional_cluster_metrics = {}
        total_distance = 0.0
        total_time = 0.0

        # Find the next available truck and robot indices
        next_truck_idx = next_route_idx % len(trucks)

        # Create a direct route for each unassigned cluster
        for cluster_id in unassigned_clusters:
            # Find the cluster object
            cluster_obj = next((c for c in clusters if c.id == cluster_id), None)
            if not cluster_obj:
                print(f"No cluster object found for ID {cluster_id}")
                continue

            # Find the centroid for this cluster
            centroid = centroids.get(cluster_id)
            if not centroid:
                # Try to find by centroid index if direct lookup fails
                centroid_idx = cluster_id_to_centroid_idx.get(cluster_id)
                if centroid_idx:
                    centroid = self._find_centroid_by_index(centroid_idx, centroids)

                if not centroid:
                    print(f"No centroid found for cluster {cluster_id}")
                    continue

            # Assign a truck for this route
            truck_idx = next_truck_idx % len(trucks)
            truck = trucks[truck_idx]
            truck_robots = robots_per_truck[truck_idx]

            # Create direct route from depot to this cluster and back
            route_idx = len(additional_truck_routes) + next_route_idx
            route = [0, cluster_id_to_centroid_idx.get(cluster_id, route_idx + 1), 0]
            additional_truck_routes.append(route)

            # Calculate route distance
            route_distance = depot.distance_to(centroid) * 2  # Round trip
            total_distance += route_distance

            # Calculate route time
            route_time = route_distance / truck.speed
            total_time += route_time

            # Route robots within this cluster
            robot_routes, robot_metrics = self.robot_routing_service.route_robots(
                cluster_centroid=centroid,
                customers=cluster_obj.locations,
                robots=truck_robots,
                truck=truck,
            )

            # Store the robot routes for this cluster
            additional_cluster_routes[cluster_id] = robot_routes

            # Track metrics for this cluster
            cluster_time = self.robot_unloading_time + robot_metrics["max_robot_time"]
            additional_cluster_metrics[route_idx] = {
                "cluster_times": [cluster_time],
                "total_robot_distance": robot_metrics["total_robot_distance"],
                "total_time": cluster_time,
                "max_time": cluster_time,  # For a single cluster, max = total
            }

            next_truck_idx += 1

        # Return the additional routes and metrics
        return {
            "truck_routes": additional_truck_routes,
            "cluster_routes": additional_cluster_routes,
            "cluster_metrics": additional_cluster_metrics,
            "total_distance": total_distance,
            "total_time": total_time,
        }

    def _find_centroid_by_index(
        self, idx: int, centroids: Dict[int, Location]
    ) -> Location:
        """
        Find a centroid by its index in route.

        Args:
            idx: Index of the centroid in route
            centroids: Dictionary mapping cluster ID to centroid location

        Returns:
            Centroid location if found, None otherwise
        """
        # If idx is the actual cluster ID
        if idx in centroids:
            return centroids[idx]

        # Otherwise, try to find by position in the list
        # This is a best-effort approach
        all_centroids = list(centroids.values())
        if 0 <= idx - 1 < len(all_centroids):  # Subtract 1 because depot is at index 0
            return all_centroids[idx - 1]

        return None
