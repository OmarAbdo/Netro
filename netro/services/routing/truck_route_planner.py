# netro/services/routing/truck_route_planner.py
from typing import List, Dict, Tuple, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck
from netro.core.interfaces.routing import RoutingAlgorithm


class TruckRoutePlanner:
    """
    Plans routes for trucks to visit cluster centroids.

    Based on:
    Ostermeier et al. (2022), "Cost-optimal truck-and-robot routing for last-mile delivery".
    Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems",
    Computers and Electrical Engineering, 67, 596–607.
    """

    def __init__(self, routing_algorithm: RoutingAlgorithm):
        """
        Initialize the truck route planner.

        Args:
            routing_algorithm: Algorithm to use for truck routing.
        """
        self.routing_algorithm = routing_algorithm

    def plan_routes(
        self, depot: Location, centroids: Dict[int, Location], trucks: List[Truck]
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Plan routes for trucks between cluster centroids.

        Args:
            depot: Depot location.
            centroids: Dictionary mapping cluster ID to centroid location.
            trucks: List of available trucks.

        Returns:
            Tuple containing:
            - List of truck routes.
            - Dictionary with metrics.
        """
        # Force solver to use all available trucks
        self.routing_algorithm.max_vehicles = len(trucks)
        
        # Create list of locations with depot as first location
        centroid_locations = list(centroids.values())
        all_locations = [depot] + centroid_locations

        print(f"Routing trucks between depot and {len(centroid_locations)} centroids")

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

        # Check if we have any centroids to route to
        if len(centroid_locations) == 0:
            print("No centroids to route to, returning empty routes")
            return [], {"total_distance": 0.0, "total_time": 0.0}

        # Check if we have any trucks
        if len(trucks) == 0:
            print("No trucks available, returning empty routes")
            return [], {"total_distance": 0.0, "total_time": 0.0}

        # Try the routing algorithm first
        routes, metrics = self._try_routing_algorithm(
            distance_matrix, demands, capacities, trucks
        )

        # If routing algorithm failed, use bin packing fallback
        if not routes or not any(len(route) > 2 for route in routes):
            print("Using bin packing fallback for truck routing")
            routes, metrics = self._bin_packing_fallback(
                distance_matrix, demands, capacities, trucks
            )

        return routes, metrics

    def _try_routing_algorithm(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        trucks: List[Truck],
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Try to solve the routing problem using the provided algorithm.

        Args:
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location.
            capacities: Array of truck capacities.
            trucks: List of trucks to use for routing.

        Returns:
            Tuple containing:
            - List of truck routes.
            - Dictionary with metrics.
        """
        try:
            routes, total_distance = self.routing_algorithm.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                capacities=capacities,
                depot_index=0,
                max_vehicles=len(trucks),
            )

            # If routes were generated, calculate metrics
            if routes and any(len(route) > 2 for route in routes):
                # Calculate truck metrics
                total_time = self._calculate_total_time(routes, distance_matrix, trucks)
                return routes, {
                    "total_distance": total_distance,
                    "total_time": total_time,
                }

        except Exception as e:
            print(f"Error solving truck routing problem: {str(e)}")

        return [], {"total_distance": 0.0, "total_time": 0.0}

    def _bin_packing_fallback(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        trucks: List[Truck],
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Use a bin-packing approach as fallback for routing.

        Args:
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location.
            capacities: Array of truck capacities.
            trucks: List of trucks to use for routing.

        Returns:
            Tuple containing:
            - List of truck routes.
            - Dictionary with metrics.
        """
        # Create a list of centroids with their demands and indices
        centroid_info = []
        for i in range(1, len(demands)):  # Skip depot (index 0)
            centroid_info.append(
                {
                    "index": i,
                    "demand": demands[i],
                }
            )

        # Sort centroids by demand (descending) for better bin packing
        centroid_info.sort(key=lambda x: x["demand"], reverse=True)

        # Create routes using a bin packing algorithm
        routes = []
        total_distance = 0.0

        # Keep track of which centroids have been assigned to a route
        assigned_centroids = set()

        # First-fit decreasing bin packing algorithm
        for truck_idx, truck in enumerate(trucks):
            if len(assigned_centroids) == len(centroid_info):
                # All centroids have been assigned
                break

            route = [0]  # Start at depot
            remaining_capacity = truck.capacity

            # Try to add centroids to this truck's route
            for centroid in centroid_info:
                if centroid["index"] in assigned_centroids:
                    # This centroid has already been assigned to a route
                    continue

                if centroid["demand"] <= remaining_capacity:
                    # This centroid can be added to the route
                    route.append(centroid["index"])
                    assigned_centroids.add(centroid["index"])
                    remaining_capacity -= centroid["demand"]

            if len(route) > 1:  # Only add route if it visits at least one centroid
                route.append(0)  # Return to depot
                routes.append(route)

                # Calculate route distance
                route_distance = sum(
                    distance_matrix[route[j]][route[j + 1]]
                    for j in range(len(route) - 1)
                )
                total_distance += route_distance

        # Calculate total time
        total_time = self._calculate_total_time(routes, distance_matrix, trucks)

        return routes, {"total_distance": total_distance, "total_time": total_time}

    def _calculate_total_time(
        self, routes: List[List[int]], distance_matrix: np.ndarray, trucks: List[Truck]
    ) -> float:
        """
        Calculate the total time for all truck routes.

        Args:
            routes: List of truck routes.
            distance_matrix: Matrix of distances between locations.
            trucks: List of trucks.

        Returns:
            Total time for all routes.
        """
        total_time = 0.0
        for i, route in enumerate(routes):
            if len(route) <= 2:  # Skip routes that only visit depot
                continue

            # Get the truck for this route
            truck = trucks[min(i, len(trucks) - 1)]

            # Calculate route distance
            route_distance = sum(
                distance_matrix[route[j]][route[j + 1]] for j in range(len(route) - 1)
            )

            # Calculate route time
            route_time = route_distance / truck.speed
            total_time += route_time

        return total_time
