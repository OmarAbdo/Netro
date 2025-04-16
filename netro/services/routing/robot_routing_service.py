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
        print(f"DEBUG: Routing robots at centroid ({cluster_centroid.x}, {cluster_centroid.y})")
        print(f"DEBUG: Number of customers: {len(customers)}")
        print(f"DEBUG: Number of robots: {len(robots)}")
        
        if not customers:
            print("DEBUG: No customers to route, returning empty routes")
            return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}
        
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
        print(f"DEBUG: Customer demands: {demands}")

        # Robot capacities
        capacities = np.array([robot.capacity for robot in robots])
        print(f"DEBUG: Robot capacities: {capacities}")

        # Ensure we have at least one robot
        if len(robots) == 0:
            print("DEBUG: No robots available, returning empty routes")
            return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}

        # Solve the routing problem
        try:
            routes, total_distance = self.routing_algorithm.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                capacities=capacities,
                depot_index=0,
                max_vehicles=len(robots),
            )
            
            print(f"DEBUG: Generated {len(routes)} robot routes with total distance {total_distance}")
            for i, route in enumerate(routes):
                print(f"DEBUG: Robot route {i}: {route}")
                
            # Check if we have valid routes
            if not routes or all(len(route) <= 2 for route in routes):  # Only depot or empty routes
                print("DEBUG: No valid robot routes generated, creating simple routes")
                
                # Create simple routes for robots
                routes = []
                remaining_customers = list(range(1, n_locations))  # Skip depot (index 0)
                robot_idx = 0
                
                # Distribute customers among robots based on capacity
                while remaining_customers and robot_idx < len(robots):
                    robot = robots[robot_idx]
                    route = [0]  # Start at depot
                    current_load = 0
                    
                    # Add customers to route until capacity is reached
                    i = 0
                    while i < len(remaining_customers):
                        customer_idx = remaining_customers[i]
                        customer_demand = demands[customer_idx]
                        
                        # If adding this customer would exceed capacity, skip it
                        if current_load + customer_demand > robot.capacity:
                            i += 1
                            continue
                            
                        # Add customer to route
                        route.append(customer_idx)
                        current_load += customer_demand
                        remaining_customers.pop(i)  # Remove from remaining customers
                        
                        # If we've reached capacity, stop adding customers
                        if current_load >= robot.capacity:
                            break
                    
                    # Return to depot
                    route.append(0)
                    
                    # Only add route if it visits at least one customer
                    if len(route) > 2:
                        routes.append(route)
                        
                        # Calculate route distance
                        route_distance = 0.0
                        for j in range(len(route) - 1):
                            route_distance += distance_matrix[route[j], route[j + 1]]
                        total_distance += route_distance
                        
                        print(f"DEBUG: Created simple route for robot {robot_idx}: {route} with distance {route_distance}")
                    
                    robot_idx += 1
                
                # If we still have customers but no more robots, create additional routes
                # by reusing robots (this is a simplification)
                # Add a safety counter to prevent infinite loops
                safety_counter = 0
                max_iterations = 100  # Limit the number of iterations to prevent infinite loops
                
                while remaining_customers and safety_counter < max_iterations:
                    safety_counter += 1
                    print(f"DEBUG: Additional route iteration {safety_counter}, {len(remaining_customers)} customers remaining")
                    
                    # Use robots in round-robin fashion
                    robot_idx = robot_idx % len(robots)
                    robot = robots[robot_idx]
                    
                    route = [0]  # Start at depot
                    current_load = 0
                    
                    # Add customers to route until capacity is reached
                    i = 0
                    customer_added = False
                    
                    # Limit the number of customers to process to prevent infinite loops
                    max_customers_to_process = min(len(remaining_customers), 20)
                    customers_processed = 0
                    
                    while i < len(remaining_customers) and current_load < robot.capacity and customers_processed < max_customers_to_process:
                        customers_processed += 1
                        customer_idx = remaining_customers[i]
                        customer_demand = demands[customer_idx]
                        
                        # If adding this customer would exceed capacity, skip it
                        if current_load + customer_demand > robot.capacity:
                            i += 1
                            continue
                            
                        # Add customer to route
                        route.append(customer_idx)
                        current_load += customer_demand
                        remaining_customers.pop(i)  # Remove from remaining customers
                        customer_added = True
                        
                        # If we've reached capacity, stop adding customers
                        if current_load >= robot.capacity:
                            break
                    
                    # If we couldn't add any customers, break to avoid infinite loop
                    if not customer_added:
                        print(f"DEBUG: No customers could be added to route, breaking loop with {len(remaining_customers)} customers remaining")
                        break
                    
                    # Return to depot
                    route.append(0)
                    
                    # Only add route if it visits at least one customer
                    if len(route) > 2:
                        routes.append(route)
                        
                        # Calculate route distance
                        route_distance = 0.0
                        for j in range(len(route) - 1):
                            route_distance += distance_matrix[route[j], route[j + 1]]
                        total_distance += route_distance
                        
                        print(f"DEBUG: Created additional route for robot {robot_idx}: {route} with distance {route_distance}")
                    
                    robot_idx += 1
                
                print(f"DEBUG: Created {len(routes)} simple robot routes with total distance {total_distance}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(routes, distance_matrix, robots)
            print(f"DEBUG: Robot metrics: {metrics}")
            
            return routes, metrics
            
        except Exception as e:
            print(f"DEBUG: Error solving robot routing problem: {str(e)}")
            
            # Create simple routes as fallback
            print("DEBUG: Creating fallback routes due to error")
            routes = []
            remaining_customers = list(range(1, n_locations))  # Skip depot (index 0)
            
            # Distribute customers among robots
            for robot_idx, robot in enumerate(robots):
                if not remaining_customers:
                    break
                    
                route = [0]  # Start at depot
                current_load = 0
                
                # Add customers to route until capacity is reached
                i = 0
                while i < len(remaining_customers):
                    customer_idx = remaining_customers[i]
                    customer_demand = demands[customer_idx]
                    
                    # If adding this customer would exceed capacity, skip it
                    if current_load + customer_demand > robot.capacity:
                        i += 1
                        continue
                        
                    # Add customer to route
                    route.append(customer_idx)
                    current_load += customer_demand
                    remaining_customers.pop(i)  # Remove from remaining customers
                    
                    # If we've reached capacity, stop adding customers
                    if current_load >= robot.capacity:
                        break
                
                # Return to depot
                route.append(0)
                
                # Only add route if it visits at least one customer
                if len(route) > 2:
                    routes.append(route)
            
            # Calculate total distance
            total_distance = 0.0
            for route in routes:
                route_distance = 0.0
                for j in range(len(route) - 1):
                    route_distance += distance_matrix[route[j], route[j + 1]]
                total_distance += route_distance
            
            # Calculate metrics
            metrics = self._calculate_metrics(routes, distance_matrix, robots)
            print(f"DEBUG: Fallback robot metrics: {metrics}")
            
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
            if not route or len(route) <= 2:  # Skip empty routes or routes with just depot
                print(f"DEBUG: Skipping empty or depot-only route {i}: {route}")
                continue

            # Get the robot for this route
            robot_idx = min(i, len(robots) - 1)
            robot = robots[robot_idx]
            print(f"DEBUG: Using robot {robot_idx} with capacity {robot.capacity} for route {i}")

            # Calculate route distance
            route_distance = 0.0
            for j in range(len(route) - 1):
                segment_distance = distance_matrix[route[j], route[j + 1]]
                route_distance += segment_distance
                print(f"DEBUG: Distance from {route[j]} to {route[j+1]}: {segment_distance}")

            total_robot_distance += route_distance
            print(f"DEBUG: Total distance for route {i}: {route_distance}")

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
                print(f"DEBUG: Robot {robot_idx} needs recharging, adding {recharge_time} time")

            # Total time for this robot's operation
            robot_time = (
                travel_time + service_time + launch_time + recovery_time + recharge_time
            )
            robot_times.append(robot_time)
            print(f"DEBUG: Total time for robot {robot_idx}: {robot_time}")

        # Metrics
        metrics = {
            "total_robot_distance": total_robot_distance,
            "max_robot_time": max(robot_times) if robot_times else 0.0,
            "total_time": sum(robot_times),
        }
        
        print(f"DEBUG: Final robot metrics: {metrics}")
        
        # Ensure we return non-zero values if we have valid routes
        if routes and any(len(route) > 2 for route in routes) and metrics["total_robot_distance"] == 0:
            print("DEBUG: Warning - zero distance despite having valid routes")

        return metrics
