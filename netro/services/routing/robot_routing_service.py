# netro/services/routing/robot_routing_service.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Robot, Truck
from netro.core.interfaces.routing import RoutingAlgorithm


class RobotRoutingService:
    """
    Service for solving the robot routing problem within a cluster.

    Based on:
    Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system",
    Transportation Research Part E.
    """

    def __init__(
        self,
        routing_algorithm: RoutingAlgorithm,
        recharge_time_factor: float = 1.5,
        robot_launch_time: float = 2.0,
        robot_recovery_time: float = 3.0,
    ):
        """
        Initialize the robot routing service.

        Args:
            routing_algorithm: Algorithm to use for robot routing.
            recharge_time_factor: Factor to compute recharge time from travel time.
            robot_launch_time: Time to launch a robot in minutes.
            robot_recovery_time: Time to recover a robot in minutes.
        """
        self.routing_algorithm = routing_algorithm
        self.recharge_time_factor = recharge_time_factor
        self.robot_launch_time = robot_launch_time
        self.robot_recovery_time = robot_recovery_time

    def route_robots(
        self,
        cluster_centroid: Location,
        customers: List[Location],
        robots: List[Robot],
        truck: Truck,
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Create routes for robots from a truck positioned at a cluster centroid.

        Args:
            cluster_centroid: Location of the cluster centroid where the truck is positioned.
            customers: List of customer locations to serve.
            robots: List of available robots.
            truck: Truck carrying the robots.

        Returns:
            A tuple containing:
            - List of robot routes (each route is a list of customer indices).
            - Dictionary with metrics: total_robot_distance, max_robot_time, total_time.
        """
        print(
            f"Routing robots at centroid ({cluster_centroid.x:.1f}, {cluster_centroid.y:.1f})"
        )
        print(f"Number of customers: {len(customers)}")
        print(f"Number of robots: {len(robots)}")

        if not customers:
            print("No customers to route, returning empty routes")
            return [], {
                "total_robot_distance": 0.0,
                "max_robot_time": 0.0,
                "total_time": 0.0,
            }

        # Ensure we have at least one robot
        if not robots:
            print("No robots available, returning empty routes")
            return [], {
                "total_robot_distance": 0.0,
                "max_robot_time": 0.0,
                "total_time": 0.0,
            }

        # Create distance matrix and prepare data for routing
        distance_matrix, demands, all_locations = self._prepare_routing_data(
            cluster_centroid, customers
        )

        # Get robot capacities
        capacities = np.array([robot.capacity for robot in robots])

        # Try to solve using the routing algorithm
        routes, metrics = self._solve_with_routing_algorithm(
            distance_matrix, demands, capacities, robots
        )

        # If no valid routes were generated, use fallback method
        if not routes or not any(len(route) > 2 for route in routes):
            print(
                "Routing algorithm failed to produce valid routes, using fallback method"
            )
            routes, metrics = self._create_fallback_routes(
                distance_matrix, demands, capacities, robots
            )

        return routes, metrics

    def _prepare_routing_data(
        self, cluster_centroid: Location, customers: List[Location]
    ) -> Tuple[np.ndarray, np.ndarray, List[Location]]:
        """
        Prepare data needed for the routing algorithm.

        Args:
            cluster_centroid: Location of the cluster centroid.
            customers: List of customer locations.

        Returns:
            Tuple containing:
            - Distance matrix
            - Demands array
            - List of all locations
        """
        # Create combined list of locations with centroid as first location (index 0)
        all_locations = [cluster_centroid] + customers

        # Compute distance matrix
        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations))

        for i in range(n_locations):
            for j in range(n_locations):
                distance_matrix[i, j] = all_locations[i].distance_to(all_locations[j])

        # Extract demands (0 for centroid)
        demands = np.array([loc.demand for loc in all_locations])

        return distance_matrix, demands, all_locations

    def _solve_with_routing_algorithm(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        robots: List[Robot],
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Solve the routing problem using the provided algorithm.

        Args:
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location.
            capacities: Array of robot capacities.
            robots: List of robots.

        Returns:
            Tuple containing:
            - List of robot routes.
            - Dictionary with metrics.
        """
        try:
            routes, total_distance = self.routing_algorithm.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                capacities=capacities,
                depot_index=0,
                max_vehicles=len(robots),
            )

            # Calculate detailed metrics
            if routes and any(len(route) > 2 for route in routes):
                metrics = self._calculate_metrics(routes, distance_matrix, robots)
                return routes, metrics

        except Exception as e:
            print(f"Error solving robot routing problem: {str(e)}")

        return [], {
            "total_robot_distance": 0.0,
            "max_robot_time": 0.0,
            "total_time": 0.0,
        }

    def _create_fallback_routes(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        robots: List[Robot],
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Create simple but REALISTIC routes as fallback when routing algorithm fails.

        KEY FIX: Distribute customers among robots for PARALLEL operation.
        """
        routes = []
        remaining_customers = list(range(1, len(demands)))  # Skip centroid (index 0)

        print(f"Fallback: {len(remaining_customers)} customers, {len(robots)} robots")

        # Simple distribution: give each robot a fair share of customers
        customers_per_robot = len(remaining_customers) // len(robots)
        extra_customers = len(remaining_customers) % len(robots)

        customer_idx = 0
        for robot_idx, robot in enumerate(robots):
            if customer_idx >= len(remaining_customers):
                break

            route = [0]  # Start at centroid
            current_load = 0
            customers_for_this_robot = customers_per_robot + (
                1 if robot_idx < extra_customers else 0
            )

            # Add customers to this robot's route up to its capacity
            customers_added = 0
            while (
                customer_idx < len(remaining_customers)
                and customers_added < customers_for_this_robot
            ):

                customer_location_idx = remaining_customers[customer_idx]
                customer_demand = demands[customer_location_idx]

                # Check capacity constraint
                if current_load + customer_demand <= robot.capacity:
                    route.append(customer_location_idx)
                    current_load += customer_demand
                    customers_added += 1
                    customer_idx += 1
                else:
                    # Skip this customer if it exceeds capacity
                    customer_idx += 1

            # Return to centroid
            route.append(0)

            # Only add route if it visits at least one customer
            if len(route) > 2:
                routes.append(route)
                print(
                    f"Robot {robot_idx}: route length={len(route)-2}, demand={current_load:.1f}/{robot.capacity}"
                )

        print(
            f"Created {len(routes)} robot routes covering {sum(len(r)-2 for r in routes)} customers"
        )

        # Calculate metrics for the created routes
        metrics = self._calculate_metrics(routes, distance_matrix, robots)
        return routes, metrics

    def _calculate_metrics(
        self, routes: List[List[int]], distance_matrix: np.ndarray, robots: List[Robot]
    ) -> Dict[str, float]:
        """
        Calculate metrics for robot routes with CORRECTED parallel operation logic.

        The key insight: robots operate in PARALLEL within a cluster.
        Driver waits for the SLOWEST robot, not the sum of all robot times.

        Args:
            routes: List of robot routes.
            distance_matrix: Distance matrix.
            robots: List of robots.

        Returns:
            Dictionary with metrics including proper parallel time calculation.
        """
        total_robot_distance = 0.0
        robot_times = []

        for i, route in enumerate(routes):
            if not route or len(route) <= 2:  # Skip empty routes
                continue

            # Get the robot for this route
            robot_idx = min(i, len(robots) - 1)
            robot = robots[robot_idx]

            # Calculate route distance
            route_distance = sum(
                distance_matrix[route[j], route[j + 1]] for j in range(len(route) - 1)
            )
            total_robot_distance += route_distance

            # Calculate route time components
            travel_time = route_distance / robot.speed  # Hours

            # Service time per customer (convert to hours)
            num_customers = len(route) - 2  # Exclude centroid start and end
            service_time = num_customers * 5.0 / 60.0  # 5 minutes per customer

            # Launch and recovery time (convert to hours)
            launch_time = self.robot_launch_time / 60.0
            recovery_time = self.robot_recovery_time / 60.0

            # Recharge time if needed
            recharge_time = 0.0
            battery_capacity_hours = robot.battery_capacity / 60.0  # Convert to hours
            if travel_time > battery_capacity_hours:
                recharge_time = (
                    travel_time - battery_capacity_hours
                ) * self.recharge_time_factor

            # Total time for this robot's complete operation
            robot_time = (
                travel_time + service_time + launch_time + recovery_time + recharge_time
            )
            robot_times.append(robot_time)

        # CORRECTED: In parallel operation, driver waits for the SLOWEST robot
        # This is the key fix - robots work in parallel!
        max_robot_time = max(robot_times) if robot_times else 0.0

        # Convert max_robot_time to minutes for cluster operations
        max_robot_time_minutes = max_robot_time * 60.0

        print(
            f"Robot metrics: individual_times={[f'{t:.2f}h' for t in robot_times]}, max_time={max_robot_time:.2f}h"
        )

        metrics = {
            "total_robot_distance": total_robot_distance,
            "max_robot_time": max_robot_time_minutes,  # Convert back to minutes for compatibility
            "total_time": sum(robot_times),  # Sequential sum for reference only
            "individual_robot_times": robot_times,
            "num_active_robots": len(robot_times),
        }

        return metrics
