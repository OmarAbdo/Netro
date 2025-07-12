# netro/services/robot_deployment/smart_loader.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Robot, Truck


class SmartLoader:
    """
    Smart Loader service for optimizing robot loading and deployment.

    Inspired by:
    Simoni et al., "Optimization and analysis of a robot-assisted last mile delivery system",
    Transportation Research Part E.
    """

    def __init__(self, loading_time_per_robot: float = 1.0):
        """
        Initialize the SmartLoader.

        Args:
            loading_time_per_robot: Time to load or unload a single robot in minutes.
        """
        self.loading_time_per_robot = loading_time_per_robot

    def optimize_loading(
        self, clusters: Dict[int, List[Location]], truck: Truck, robots: List[Robot]
    ) -> Dict[str, Any]:
        """
        Optimize the loading of robots onto the truck based on cluster assignments.

        Uses a First-Fit Decreasing (FFD) bin packing algorithm to assign orders to robots.

        Args:
            clusters: Dictionary mapping cluster IDs to lists of customer locations.
            truck: The truck carrying the robots.
            robots: List of available robots.

        Returns:
            Dictionary with loading plan details.
        """
        loading_plan = {}
        cluster_robot_assignments = {}

        # For each cluster, assign orders to robots
        for cluster_id, locations in clusters.items():
            # Sort locations by demand (descending)
            sorted_locations = sorted(
                locations, key=lambda loc: loc.demand, reverse=True
            )

            # Calculate how many robots we need for this cluster
            cluster_robot_assignments[cluster_id] = self._assign_robots_to_cluster(
                sorted_locations, robots
            )

        # Calculate loading time
        total_robots_used = sum(
            len(assignments) for assignments in cluster_robot_assignments.values()
        )
        loading_time = total_robots_used * self.loading_time_per_robot

        # Create final loading plan
        loading_plan = {
            "cluster_robot_assignments": cluster_robot_assignments,
            "total_robots_used": total_robots_used,
            "loading_time": loading_time,
        }

        return loading_plan

    def _assign_robots_to_cluster(
        self, locations: List[Location], robots: List[Robot]
    ) -> List[Dict[str, Any]]:
        """
        Assign customer locations to robots within a cluster.

        Args:
            locations: List of customer locations, sorted by demand (descending).
            robots: List of available robots.

        Returns:
            List of robot assignments, each containing robot info and assigned locations.
        """
        if not locations:
            return []

        if not robots:
            raise ValueError("No robots available for assignment")

        # For uniform robot capacity, use First-Fit Decreasing bin packing
        robot_assignments = []
        current_robot = {
            "robot_id": 0,
            "assigned_locations": [],
            "remaining_capacity": robots[0].capacity,
            "total_assigned_demand": 0.0,
        }

        robot_counter = 0

        for location in locations:
            # If the current robot can't handle this location, create a new robot assignment
            if location.demand > current_robot["remaining_capacity"]:
                if current_robot["assigned_locations"]:
                    robot_assignments.append(current_robot)

                robot_counter += 1
                if robot_counter >= len(robots):
                    # No more robots available, this is an error condition
                    raise ValueError(
                        f"Not enough robots to serve all customers in cluster"
                    )

                current_robot = {
                    "robot_id": robot_counter,
                    "assigned_locations": [],
                    "remaining_capacity": robots[robot_counter].capacity,
                    "total_assigned_demand": 0.0,
                }

            # Assign the location to the current robot
            current_robot["assigned_locations"].append(location)
            current_robot["remaining_capacity"] -= location.demand
            current_robot["total_assigned_demand"] += location.demand

        # Add the last robot if it has any assignments
        if current_robot["assigned_locations"]:
            robot_assignments.append(current_robot)

        return robot_assignments
