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

        # Robot capacities
        capacities = np.array([robot.capacity for robot in robots])

        # Solve the routing problem
        routes, total_distance = self.routing_algorithm.solve(
            distance_matrix=distance_matrix,
            demands=demands,
            capacities=capacities,
            depot_index=0,
            max_vehicles=len(robots),
        )

        # Calculate metrics
        metrics = self._calculate_metrics(routes, distance_matrix, robots)

        return routes, metrics

    def _calculate_metrics(
        self, routes: List[List[int]], distance_matrix: np.ndarray, robots: List[Robot]
    ) -> Dict[str, float]:
        """
        Calculate metrics for robot routes.

        Args:
            routes: List of robot routes.
            distance_matrix: Distance matrix.
            robots: List of robots.

        Returns:
            Dictionary with metrics.
        """
        total_robot_distance = 0.0
        robot_times = []

        for i, route in enumerate(routes):
            if not route:
                continue

            # Get the robot for this route
            robot = robots[min(i, len(robots) - 1)]

            # Calculate route distance
            route_distance = 0.0
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]

            total_robot_distance += route_distance

            # Calculate route time
            travel_time = route_distance / robot.speed
            service_time = 0.0  # Add customer service times if available
            launch_time = self.robot_launch_time
            recovery_time = self.robot_recovery_time
            recharge_time = 0.0

            # Add recharge time if needed based on battery consumption
            if travel_time > robot.battery_capacity:
                recharge_time = (
                    travel_time - robot.battery_capacity
                ) * self.recharge_time_factor

            # Total time for this robot's operation
            robot_time = (
                travel_time + service_time + launch_time + recovery_time + recharge_time
            )
            robot_times.append(robot_time)

        # Metrics
        metrics = {
            "total_robot_distance": total_robot_distance,
            "max_robot_time": max(robot_times) if robot_times else 0.0,
            "total_time": sum(robot_times),
        }

        return metrics
