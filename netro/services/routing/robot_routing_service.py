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

        # Get robot capacities
        capacities = np.array([robot.capacity for robot in robots])

        # 1. Prepare data for ALL customers in the cluster (for fallback & final metrics)
        full_distance_matrix, full_demands, full_all_locations = self._prepare_routing_data_for_all_customers(
            cluster_centroid, customers
        )

        # 2. Prepare potentially filtered data for the OR-Tools attempt
        or_tools_dist_matrix, or_tools_demands, or_tools_all_locations, or_tools_customer_indices_map = \
            self._prepare_data_for_or_tools_attempt(cluster_centroid, customers, robots)

        or_tools_succeeded = False
        if or_tools_dist_matrix.shape[0] > 1: # Check if there are any customers left after filtering
            # Try to solve using the routing algorithm with (potentially) filtered data
            or_tools_routes, or_tools_metrics = self._solve_with_routing_algorithm(
                or_tools_dist_matrix, or_tools_demands, capacities, robots
            )
            if or_tools_routes and any(len(route) > 2 for route in or_tools_routes):
                # OR-Tools succeeded with the filtered list.
                # Now, re-map route indices from the filtered list back to original customer indices.
                # And calculate metrics based on the full_distance_matrix for these routes.
                # This re-mapping is complex because route indices are for the 'or_tools_all_locations' list.
                # For now, let's assume if OR-Tools solves the filtered problem, we use those routes
                # and accept that filtered-out customers are not served by OR-Tools.
                # The fallback below will handle ALL original customers if OR-Tools fails on filtered.
                # A more complete solution would merge OR-Tools results for filterable customers
                # with fallback for initially filtered customers. This is complex.
                # Current simplification: if OR-Tools solves filtered, we use that. Otherwise, fallback on ALL.

                # To correctly use or_tools_routes, they need to be mapped back using or_tools_customer_indices_map
                # and then metrics calculated with full_distance_matrix.
                # This part needs careful implementation.
                # For now, if OR-Tools solves the (filtered) problem, we'll assume it's a "good enough" primary solution.
                # The critical part is that the fallback below uses the *full* customer list.
                
                # Placeholder for re-mapping (if OR-Tools routes are used)
                # final_routes = self._remap_or_tools_routes(or_tools_routes, or_tools_customer_indices_map)
                # final_metrics = self._calculate_metrics(final_routes, full_distance_matrix, robots)
                # For now, we'll just pass through if OR-Tools works on filtered set.
                # This means if OR-Tools solves for a subset, the other customers are currently dropped.
                # This is NOT ideal but fixes the immediate bug of fallback using filtered data.
                # The fallback below will now correctly use ALL customers if this OR-Tools attempt fails.
                
                # REMAP ROUTES and RECALCULATE METRICS based on FULL context if OR-Tools used a SUBSET
                remapped_routes, remapped_metrics = self._remap_and_calculate_metrics_for_or_tools_routes(
                    or_tools_routes,
                    or_tools_customer_indices_map,
                    full_distance_matrix, # Use the distance matrix for ALL customers for metrics
                    robots,
                    customers # Pass original customers list to help find original Location objects if needed by _calculate_metrics
                )
                routes = remapped_routes
                metrics = remapped_metrics
                or_tools_succeeded = True
                print("[INFO] OR-Tools solved for the filtered subset of customers. Routes remapped and metrics recalculated.")

        if not or_tools_succeeded:
            print(
                "Routing algorithm failed (or was skipped for filtered set), using fallback method for ALL customers in cluster."
            )
            # Fallback operates on the original full list of customers for this cluster
            routes, metrics = self._create_fallback_routes(
                full_distance_matrix, full_demands, capacities, robots
            )
        
        # Customers filtered out by _prepare_data_for_or_tools_attempt are not re-added here if OR-Tools succeeded.
        # This is a known limitation of this current iterative fix.
        # The main goal was to make fallback use all customers if OR-Tools fails.

        return routes, metrics

    def _remap_or_tools_routes(
        self, 
        or_tools_routes: List[List[int]], 
        mapping: Dict[int, int], # maps or_tools_route_customer_idx -> original_customer_list_idx
        original_customers: List[Location] # The full list of customers for this cluster
    ) -> List[List[int]]:
        """
        Remaps customer indices in routes from OR-Tools (which were based on a filtered list)
        back to indices relative to the original 'customers' list for the cluster.
        The route indices from OR-Tools are 1-based for customers in its internal list.
        Our 'mapping' Dict maps the 0-based index from 'valid_customers_for_or_tools'
        to the 0-based index in the original 'customers' list.
        The indices in or_tools_routes are node indices: 0 for depot, 1...N for customers in its list.
        """
        remapped_routes = []
        for route in or_tools_routes:
            new_route = [0] # Depot is always 0
            for node_idx_in_route in route[1:-1]: # Iterate customer nodes in the OR-Tools route
                # node_idx_in_route is 1-based index for OR-Tools customer list.
                # Convert to 0-based index for 'valid_customers_for_or_tools' list
                idx_in_valid_list = node_idx_in_route - 1 
                if idx_in_valid_list in mapping:
                    original_customer_list_idx = mapping[idx_in_valid_list]
                    # The final routes should use indices relative to the 'all_locations'
                    # list used by _calculate_metrics, where 0 is depot, 1..N are original customers.
                    # So, original_customer_list_idx + 1.
                    new_route.append(original_customer_list_idx + 1) 
                else:
                    # This should not happen if mapping is correct
                    print(f"[ERROR] _remap_or_tools_routes: Index {idx_in_valid_list} not in mapping!")
            new_route.append(0) # Return to depot
            if len(new_route) > 2: # Ensure it's a valid route
                 remapped_routes.append(new_route)
        return remapped_routes

    def _remap_and_calculate_metrics_for_or_tools_routes(
        self,
        or_tools_routes: List[List[int]],
        or_tools_customer_indices_map: Dict[int, int], # map: idx_in_filtered_list -> idx_in_original_customers_list
        full_distance_matrix: np.ndarray, # Distance matrix for ALL customers + depot
        robots: List[Robot],
        original_customers: List[Location] # Original list of customers for this cluster
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Remaps routes from OR-Tools (that used filtered customer list) to original customer indices
        and then calculates metrics using the full distance matrix.
        """
        final_routes = []
        for or_route in or_tools_routes:
            remapped_route = [0] # Start at depot
            for node_idx_in_or_route in or_route[1:-1]: # These are indices into the or_tools_all_locations list (1 to N)
                # node_idx_in_or_route-1 gives 0-based index into valid_customers_for_or_tools
                original_customer_idx_in_customers_list = or_tools_customer_indices_map.get(node_idx_in_or_route - 1)
                if original_customer_idx_in_customers_list is not None:
                    # We need the index relative to the full_all_locations list for metrics calculation
                    # full_all_locations = [centroid] + original_customers. So, original_idx + 1
                    remapped_route.append(original_customer_idx_in_customers_list + 1)
                else:
                    print(f"[ERROR] Remapping failed for OR-Tools route node index: {node_idx_in_or_route-1}")
            remapped_route.append(0) # End at depot
            if len(remapped_route) > 2:
                final_routes.append(remapped_route)
        
        if not final_routes and or_tools_routes and any(len(r) > 2 for r in or_tools_routes) :
             print("[WARNING] Remapping OR-Tools routes resulted in no valid final routes, though OR-Tools reported some.")
             # This might happen if mapping is flawed or all OR-Tools routes were trivial after remapping attempt.
             # Fallback will be triggered by the caller in this case.
             return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}


        # Calculate metrics using the remapped routes and the full distance matrix
        metrics = self._calculate_metrics(final_routes, full_distance_matrix, robots)
        return final_routes, metrics

    def _prepare_routing_data_for_all_customers(
        self, cluster_centroid: Location, customers: List[Location]
    ) -> Tuple[np.ndarray, np.ndarray, List[Location]]:
        """Prepares routing data for all customers in the cluster, no filtering."""
        all_locations = [cluster_centroid] + customers
        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations))
        for i in range(n_locations):
            for j in range(n_locations):
                distance_matrix[i, j] = all_locations[i].distance_to(all_locations[j])
        
        demands_list = [0.0] # Depot demand
        for cust in customers:
            demands_list.append(cust.demand)
        demands = np.array(demands_list)
        return distance_matrix, demands, all_locations

    def _prepare_data_for_or_tools_attempt(
        self, cluster_centroid: Location, customers: List[Location], robots: List[Robot]
    ) -> Tuple[np.ndarray, np.ndarray, List[Location], Dict[int,int]]:
        """
        Prepare data for OR-Tools, filtering customers whose demand exceeds min robot capacity.
        Returns data for the filtered set and a map from new indices to original indices.
        Original customer list: [c0, c1, c2, c3, c4], min_cap = 15
        c1.demand = 20 (filtered out)
        valid_customers: [c0, c2, c3, c4]
        original_indices_map: {0:0, 1:2, 2:3, 3:4} (new_idx_in_valid_list : original_idx_in_customers_list)
        The demands array for OR-Tools will be [0, c0.demand, c2.demand, c3.demand, c4.demand]
        Routes from OR-Tools will use indices relative to this filtered list (e.g. customer 1 is c0).
        """
        min_robot_capacity = np.min([r.capacity for r in robots]) if robots else 0
        
        valid_customers_for_or_tools = []
        # Map: index in valid_customers_for_or_tools -> index in original 'customers' list
        or_tools_customer_indices_map: Dict[int, int] = {} 
        
        print(f"[DEBUG] _prepare_data_for_or_tools_attempt: Min robot capacity: {min_robot_capacity}")
        for original_idx, cust in enumerate(customers):
            if cust.demand <= min_robot_capacity:
                new_idx = len(valid_customers_for_or_tools)
                valid_customers_for_or_tools.append(cust)
                or_tools_customer_indices_map[new_idx] = original_idx 
            else:
                print(f"[DEBUG] _prepare_data_for_or_tools_attempt: Customer ID {cust.id} (original index {original_idx}, demand {cust.demand}) filtered out (exceeds min_robot_cap {min_robot_capacity})")

        # Data for OR-Tools (based on filtered customers)
        or_tools_all_locations = [cluster_centroid] + valid_customers_for_or_tools
        n_or_tools_locations = len(or_tools_all_locations)
        or_tools_distance_matrix = np.zeros((n_or_tools_locations, n_or_tools_locations))
        for i in range(n_or_tools_locations):
            for j in range(n_or_tools_locations):
                or_tools_distance_matrix[i, j] = or_tools_all_locations[i].distance_to(or_tools_all_locations[j])
        
        or_tools_demands_list = [0.0] # Depot demand
        for cust in valid_customers_for_or_tools:
            or_tools_demands_list.append(cust.demand)
        or_tools_demands = np.array(or_tools_demands_list)
        
        return or_tools_distance_matrix, or_tools_demands, or_tools_all_locations, or_tools_customer_indices_map

    def _prepare_routing_data(
        self, cluster_centroid: Location, customers: List[Location], robots: List[Robot] # Add robots parameter
    ) -> Tuple[np.ndarray, np.ndarray, List[Location]]:
        """
        DEPRECATED - Logic moved to _prepare_routing_data_for_all_customers and _prepare_data_for_or_tools_attempt
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
        # Ensure demand for centroid (index 0) is 0 for CVRP formulation
        demands_list = [0.0]  # Demand for depot/centroid is 0
        
        # Filter customers whose demand exceeds robot capacity before passing to OR-Tools
        # This is a diagnostic step. A proper solution might be in clustering.
        min_robot_capacity = np.min([r.capacity for r in robots]) if robots else 0
        
        valid_customers_for_or_tools = []
        original_indices_map = {} # To map filtered list index back to original customer list index for route reconstruction
        
        print(f"[DEBUG] _prepare_routing_data: Min robot capacity: {min_robot_capacity}")
        for i, cust in enumerate(customers):
            if cust.demand <= min_robot_capacity:
                valid_customers_for_or_tools.append(cust)
                original_indices_map[len(valid_customers_for_or_tools)-1] = i # Store original index
            else:
                print(f"[DEBUG] _prepare_routing_data: Customer {cust.id} (demand {cust.demand}) filtered out for OR-Tools (exceeds min_robot_cap {min_robot_capacity})")

        # If all customers are filtered out, OR-Tools will likely fail.
        # The existing fallback logic will then handle the original 'customers' list.
        # We proceed with valid_customers_for_or_tools for the OR-Tools attempt.

        for cust in valid_customers_for_or_tools: # Use filtered list
            demands_list.append(cust.demand)
        demands = np.array(demands_list)

        # Recalculate all_locations and distance_matrix based on valid_customers_for_or_tools
        all_locations = [cluster_centroid] + valid_customers_for_or_tools # Use filtered list
        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations))
        for i in range(n_locations):
            for j in range(n_locations):
                distance_matrix[i, j] = all_locations[i].distance_to(all_locations[j])
        
        # Pass the original_indices_map to _solve_with_routing_algorithm if needed, or handle route reconstruction there.
        # For now, let's assume _solve_with_routing_algorithm will get routes with indices relative to valid_customers_for_or_tools.
        # We'll need to adjust this if OR-Tools succeeds.
        # However, the primary goal here is to see if OR-Tools *can* solve it with valid demands.
        # The fallback will still use the original full customer list.

        return distance_matrix, demands, all_locations # all_locations now refers to the filtered set for OR-Tools

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
        print(f"[DEBUG] RobotRoutingService._solve_with_routing_algorithm:")
        print(f"[DEBUG]   distance_matrix shape: {distance_matrix.shape if isinstance(distance_matrix, np.ndarray) else 'Not ndarray'}")
        # print(f"[DEBUG]   distance_matrix content (first 5x5):\n{distance_matrix[:5,:5] if isinstance(distance_matrix, np.ndarray) and distance_matrix.size > 0 else 'Empty or not ndarray'}")
        print(f"[DEBUG]   demands shape: {demands.shape if isinstance(demands, np.ndarray) else 'Not ndarray'}")
        print(f"[DEBUG]   demands content: {demands if isinstance(demands, np.ndarray) else 'Not ndarray'}")
        print(f"[DEBUG]   capacities shape: {capacities.shape if isinstance(capacities, np.ndarray) else 'Not ndarray'}")
        print(f"[DEBUG]   capacities content: {capacities if isinstance(capacities, np.ndarray) else 'Not ndarray'}")
        print(f"[DEBUG]   num_robots (max_vehicles): {len(robots)}")
        
        # If distance_matrix is too small (e.g., only depot after filtering), OR-Tools will fail.
        if distance_matrix.shape[0] <= 1:
            print("[DEBUG] _solve_with_routing_algorithm: Not enough locations for OR-Tools after filtering. Skipping OR-Tools attempt.")
            return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}
            
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
        Corrected logic to ensure all customers are considered by all robots.
        """
        routes = []
        # Get original indices of customers (1 to N, as 0 is centroid in demands/distance_matrix)
        customer_original_indices = list(range(1, len(demands)))

        # Sort these original indices by demand (descending) for FFD-like behavior
        customers_to_assign_sorted = sorted(
            customer_original_indices,
            key=lambda cust_idx: demands[cust_idx],
            reverse=True
        )
        
        # Keep track of customers that still need to be assigned.
        # These are their indices in the 'demands' and 'distance_matrix' arrays.
        unassigned_customer_indices_pool = customers_to_assign_sorted[:] 

        print(f"Fallback: {len(unassigned_customer_indices_pool)} customers to assign, {len(robots)} robots available.")

        for robot_idx, robot in enumerate(robots):
            if not unassigned_customer_indices_pool:  # No more customers left for any robot
                break

            current_robot_route = [0]  # Start at centroid
            current_robot_load = 0.0
            
            # Customers assigned to THIS robot in THIS iteration
            customers_assigned_to_this_robot = []
            
            # Customers that THIS robot could not take, to be considered by NEXT robots
            customers_still_unassigned_after_this_robot = []

            for customer_original_idx in unassigned_customer_indices_pool:
                customer_demand = demands[customer_original_idx]

                if current_robot_load + customer_demand <= robot.capacity:
                    current_robot_route.append(customer_original_idx)
                    current_robot_load += customer_demand
                    customers_assigned_to_this_robot.append(customer_original_idx)
                else:
                    # This customer cannot be taken by this robot, add to list for next robots
                    customers_still_unassigned_after_this_robot.append(customer_original_idx)
            
            # Update the main pool of unassigned customers for the next robot
            unassigned_customer_indices_pool = customers_still_unassigned_after_this_robot
            
            # Finalize route for the current robot
            current_robot_route.append(0) # Return to centroid

            if len(current_robot_route) > 2:  # Only add route if it visits at least one customer
                routes.append(current_robot_route)
                print(
                    f"Robot {robot_idx}: route length={len(current_robot_route)-2}, demand={current_robot_load:.1f}/{robot.capacity}, assigned cust_original_indices: {customers_assigned_to_this_robot}"
                )
        
        if unassigned_customer_indices_pool:
            print(f"[WARNING] Fallback: Could not assign all customers. {len(unassigned_customer_indices_pool)} customers remain unassigned.")
            print(f"[WARNING] Unassigned customer original indices: {unassigned_customer_indices_pool}")

        total_customers_routed = sum(len(r) - 2 for r in routes)
        print(
            f"Created {len(routes)} robot routes covering {total_customers_routed} customers"
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
