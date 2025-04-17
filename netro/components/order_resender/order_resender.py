# netro/components/order_resender/order_resender.py
from typing import List, Dict, Any, Set, Optional
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck


class OrderResender:
    """
    Handles failed deliveries and outlier orders that couldn't be delivered by robots.

    This component identifies orders that:
    1. Were not assigned to any cluster due to location outliers
    2. Failed delivery during robot operation (battery limitations, access issues)
    3. Exceeded robot capacity constraints

    And creates direct truck routes to handle these special cases.

    Based on:
    Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system",
    Transportation Research Part E, p. 69 (lines 5-15).
    """

    def __init__(self, max_robot_travel_distance: float = 5.0):
        """
        Initialize the OrderResender.

        Args:
            max_robot_travel_distance: Maximum distance (km) a robot can travel from centroid.
                Orders beyond this distance are considered outliers.
        """
        self.max_robot_travel_distance = max_robot_travel_distance

    def identify_outliers(
        self,
        all_customers: List[Location],
        assigned_customers: Set[int],
        centroids: List[Location],
    ) -> List[Location]:
        """
        Identify customer locations that are outliers (too far from any centroid).

        Args:
            all_customers: List of all customer locations.
            assigned_customers: Set of customer IDs that have been assigned to clusters.
            centroids: List of cluster centroid locations.

        Returns:
            List of outlier customer locations.
        """
        outliers = []

        for customer in all_customers:
            # Skip if already assigned
            if customer.id in assigned_customers:
                continue

            # Check if customer is too far from any centroid
            min_distance = float("inf")
            for centroid in centroids:
                distance = customer.distance_to(centroid)
                min_distance = min(min_distance, distance)

            if min_distance > self.max_robot_travel_distance:
                outliers.append(customer)

        return outliers

    def identify_failed_deliveries(
        self,
        robot_routes: Dict[int, List[List[int]]],
        all_locations: Dict[int, Location],
    ) -> List[Location]:
        """
        Identify customer locations that failed delivery by robots.

        Args:
            robot_routes: Dictionary mapping cluster IDs to lists of robot routes.
            all_locations: Dictionary mapping location IDs to Location objects.

        Returns:
            List of customer locations with failed deliveries.
        """
        # In a real system, this would track actual delivery failures
        # For now, we simulate by checking battery constraints

        failed_deliveries = []

        # For each cluster and its robot routes
        for cluster_id, routes in robot_routes.items():
            for route in routes:
                # Skip routes with only depot/centroid
                if len(route) <= 2:
                    continue

                # Centroid is always at index 0
                centroid_idx = route[0]
                centroid = all_locations.get(centroid_idx)

                if not centroid:
                    continue

                # Check each customer in the route
                for customer_idx in route[1:-1]:  # Skip centroid at start and end
                    customer = all_locations.get(customer_idx)

                    if not customer:
                        continue

                    # Check if distance exceeds battery capacity (simplified check)
                    distance_to_centroid = customer.distance_to(centroid)

                    if distance_to_centroid > self.max_robot_travel_distance:
                        failed_deliveries.append(customer)

        return failed_deliveries

    def create_special_routes(
        self,
        outliers: List[Location],
        failed_deliveries: List[Location],
        depot: Location,
        available_trucks: List[Truck],
    ) -> Dict[str, Any]:
        """
        Create special direct routes for outliers and failed deliveries.

        Args:
            outliers: List of outlier customer locations.
            failed_deliveries: List of customer locations with failed deliveries.
            depot: Depot location.
            available_trucks: List of available trucks.

        Returns:
            Dictionary containing:
            - special_routes: List of routes for special deliveries.
            - total_distance: Total distance of all special routes.
            - total_time: Total time of all special routes.
        """
        # Combine all special case customers
        special_customers = outliers + failed_deliveries

        if not special_customers or not available_trucks:
            return {"special_routes": [], "total_distance": 0.0, "total_time": 0.0}

        # Sort customers by distance from depot (ascending)
        special_customers.sort(key=lambda c: depot.distance_to(c))

        # Create routes by assigning customers to trucks (simple greedy approach)
        special_routes = []
        total_distance = 0.0
        total_time = 0.0

        # Use truck capacity as constraint
        current_route = [0]  # Start with depot (assumed to be at index 0)
        current_load = 0

        for i, customer in enumerate(special_customers):
            # If adding this customer would exceed capacity, create a new route
            if current_load + customer.demand > available_trucks[0].capacity:
                if (
                    len(current_route) > 1
                ):  # Only add route if it visits at least one customer
                    current_route.append(0)  # Return to depot
                    special_routes.append(current_route)

                # Start a new route
                current_route = [
                    0,
                    i + 1,
                ]  # Depot + customer index (adjusted for depot)
                current_load = customer.demand
            else:
                # Add customer to current route
                current_route.append(i + 1)  # Adjusted for depot index
                current_load += customer.demand

        # Add the last route if it's not empty
        if len(current_route) > 1:
            current_route.append(0)  # Return to depot
            special_routes.append(current_route)

        # Calculate metrics
        for route in special_routes:
            route_distance = 0.0
            for i in range(len(route) - 1):
                from_idx = route[i]
                to_idx = route[i + 1]

                # Convert indices to locations
                from_loc = depot if from_idx == 0 else special_customers[from_idx - 1]
                to_loc = depot if to_idx == 0 else special_customers[to_idx - 1]

                route_distance += from_loc.distance_to(to_loc)

            total_distance += route_distance
            # Calculate time using truck speed
            route_time = route_distance / available_trucks[0].speed
            total_time += route_time

        return {
            "special_routes": special_routes,
            "total_distance": total_distance,
            "total_time": total_time,
            "special_customers": special_customers,  # Include for reference
        }
