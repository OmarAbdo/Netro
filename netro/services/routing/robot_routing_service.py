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
        customers: List[Location], # This is the original list of Location objects for this cluster
        robots: List[Robot],
        truck: Truck,
    ) -> Tuple[List[List[int]], Dict[str, float], List[Location]]:
        """
        Create routes for robots from a truck positioned at a cluster centroid.

        Args:
            cluster_centroid: Location of the cluster centroid where the truck is positioned.
            customers: List of customer locations to serve.
            robots: List of available robots.
            truck: Truck carrying the robots.

        Returns:
            A tuple containing:
            - List of robot routes (each route is a list of customer indices relative to [centroid] + customers).
            - Dictionary with metrics: total_robot_distance, max_robot_time, total_time.
            - List of Location objects for customers in this cluster that were not served by robots.
        """
        print(
            f"Routing robots at centroid ({cluster_centroid.x:.1f}, {cluster_centroid.y:.1f})"
        )
        print(f"Number of customers: {len(customers)}")
        print(f"Number of robots: {len(robots)}")

        unserved_robot_customers: List[Location] = []
        routes: List[List[int]] = []
        metrics: Dict[str, float] = {
            "total_robot_distance": 0.0,
            "max_robot_time": 0.0,
            "total_time": 0.0,
        }

        if not customers:
            print("No customers to route, returning empty routes")
            return routes, metrics, unserved_robot_customers

        if not robots:
            print("No robots available, returning empty routes")
            # All customers in this cluster are unserved by robots if no robots are available
            unserved_robot_customers = list(customers)
            return routes, metrics, unserved_robot_customers

        capacities = np.array([robot.capacity for robot in robots])

        full_distance_matrix, full_demands, full_all_locations = (
            self._prepare_routing_data_for_all_customers(cluster_centroid, customers)
        )

        (
            or_tools_dist_matrix,
            or_tools_demands,
            or_tools_all_locations_for_or_tools, # List of Location objects for OR-Tools (centroid + filtered customers)
            or_tools_customer_indices_map, # Map: idx_in_filtered_list -> idx_in_original_customers_list
        ) = self._prepare_data_for_or_tools_attempt(cluster_centroid, customers, robots)

        or_tools_succeeded = False
        if or_tools_dist_matrix.shape[0] > 1:
            or_tools_routes_raw, or_tools_metrics_raw = self._solve_with_routing_algorithm(
                or_tools_dist_matrix, or_tools_demands, capacities, robots
            )
            if or_tools_routes_raw and any(len(route) > 2 for route in or_tools_routes_raw):
                routes, metrics = (
                    self._remap_and_calculate_metrics_for_or_tools_routes(
                        or_tools_routes_raw,
                        or_tools_customer_indices_map,
                        full_distance_matrix,
                        robots,
                        customers 
                    )
                )
                or_tools_succeeded = True
                print(
                    "[INFO] OR-Tools solved for the filtered subset of customers. Routes remapped and metrics recalculated."
                )

        if not or_tools_succeeded:
            print(
                "Routing algorithm failed (or was skipped for filtered set), using fallback method for ALL customers in cluster."
            )
            routes, metrics, unassigned_indices_from_fallback = self._create_fallback_routes(
                full_distance_matrix, full_demands, capacities, robots
            )
            # Populate unserved_robot_customers from fallback
            for unassigned_idx in unassigned_indices_from_fallback: # These are 1-based indices for full_all_locations
                if 0 <= unassigned_idx - 1 < len(customers):
                    unserved_robot_customers.append(customers[unassigned_idx - 1])
        else:
            # OR-Tools succeeded, determine unserved customers
            # 1. Customers initially filtered out
            initially_filtered_out_ids = set(c.id for c in customers) - set(c.id for c in or_tools_all_locations_for_or_tools[1:])
            for cust_obj in customers:
                if cust_obj.id in initially_filtered_out_ids:
                    unserved_robot_customers.append(cust_obj)
            
            # 2. Customers from the OR-Tools set that were not in the final routes
            served_in_or_tools_routes_ids = set()
            for route in routes: # 'routes' are the remapped_routes
                for node_idx in route[1:-1]: # node_idx is 1-based index into full_all_locations ([centroid] + original customers)
                    if 0 <= node_idx - 1 < len(customers): 
                        served_in_or_tools_routes_ids.add(customers[node_idx-1].id)
            
            for cust_obj_in_or_tools_set in or_tools_all_locations_for_or_tools[1:]: # Iterate filtered customers
                if cust_obj_in_or_tools_set.id not in served_in_or_tools_routes_ids:
                    # Find the original Location object to ensure consistency if not already added
                    original_cust_obj = next((c for c in customers if c.id == cust_obj_in_or_tools_set.id), None)
                    if original_cust_obj and not any(u.id == original_cust_obj.id for u in unserved_robot_customers):
                         unserved_robot_customers.append(original_cust_obj)
        
        if unserved_robot_customers:
            print(f"[RobotRoutingService] Identified {len(unserved_robot_customers)} customers unserved within this cluster.")

        return routes, metrics, unserved_robot_customers

    def _remap_or_tools_routes(
        self,
        or_tools_routes: List[List[int]],
        mapping: Dict[
            int, int
        ],  # maps or_tools_route_customer_idx -> original_customer_list_idx
        original_customers: List[
            Location
        ],  # The full list of customers for this cluster
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
            new_route = [0]  # Depot is always 0
            for node_idx_in_route in route[
                1:-1
            ]:  # Iterate customer nodes in the OR-Tools route
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
                    print(
                        f"[ERROR] _remap_or_tools_routes: Index {idx_in_valid_list} not in mapping!"
                    )
            new_route.append(0)  # Return to depot
            if len(new_route) > 2:  # Ensure it's a valid route
                remapped_routes.append(new_route)
        return remapped_routes

    def _remap_and_calculate_metrics_for_or_tools_routes(
        self,
        or_tools_routes: List[List[int]],
        or_tools_customer_indices_map: Dict[
            int, int
        ],  # map: idx_in_filtered_list -> idx_in_original_customers_list
        full_distance_matrix: np.ndarray,  # Distance matrix for ALL customers + depot
        robots: List[Robot],
        original_customers: List[
            Location
        ],  # Original list of customers for this cluster
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Remaps routes from OR-Tools (that used filtered customer list) to original customer indices
        and then calculates metrics using the full distance matrix.
        """
        final_routes = []
        for or_route in or_tools_routes:
            remapped_route = [0]  # Start at depot
            for node_idx_in_or_route in or_route[
                1:-1
            ]:  # These are indices into the or_tools_all_locations list (1 to N)
                # node_idx_in_or_route-1 gives 0-based index into valid_customers_for_or_tools
                original_customer_idx_in_customers_list = (
                    or_tools_customer_indices_map.get(node_idx_in_or_route - 1)
                )
                if original_customer_idx_in_customers_list is not None:
                    # We need the index relative to the full_all_locations list for metrics calculation
                    # full_all_locations = [centroid] + original_customers. So, original_idx + 1
                    remapped_route.append(original_customer_idx_in_customers_list + 1)
                else:
                    print(
                        f"[ERROR] Remapping failed for OR-Tools route node index: {node_idx_in_or_route-1}"
                    )
            remapped_route.append(0)  # End at depot
            if len(remapped_route) > 2:
                final_routes.append(remapped_route)

        if (
            not final_routes
            and or_tools_routes
            and any(len(r) > 2 for r in or_tools_routes)
        ):
            print(
                "[WARNING] Remapping OR-Tools routes resulted in no valid final routes, though OR-Tools reported some."
            )
            return [], {
                "total_robot_distance": 0.0,
                "max_robot_time": 0.0,
                "total_time": 0.0,
            }

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

        demands_list = [0.0]  # Depot demand
        for cust in customers:
            demands_list.append(cust.demand)
        demands = np.array(demands_list)
        return distance_matrix, demands, all_locations

    def _prepare_data_for_or_tools_attempt(
        self, cluster_centroid: Location, customers: List[Location], robots: List[Robot]
    ) -> Tuple[np.ndarray, np.ndarray, List[Location], Dict[int, int]]:
        min_robot_capacity = np.min([r.capacity for r in robots]) if robots else 0
        valid_customers_for_or_tools = []
        or_tools_customer_indices_map: Dict[int, int] = {}
        for original_idx, cust in enumerate(customers):
            if cust.demand <= min_robot_capacity:
                new_idx = len(valid_customers_for_or_tools)
                valid_customers_for_or_tools.append(cust)
                or_tools_customer_indices_map[new_idx] = original_idx
            else:
                print(
                    f"[DEBUG] _prepare_data_for_or_tools_attempt: Customer ID {cust.id} (original index {original_idx}, demand {cust.demand}) filtered out (exceeds min_robot_cap {min_robot_capacity})"
                )
        or_tools_all_locations = [cluster_centroid] + valid_customers_for_or_tools
        n_or_tools_locations = len(or_tools_all_locations)
        or_tools_distance_matrix = np.zeros(
            (n_or_tools_locations, n_or_tools_locations)
        )
        for i in range(n_or_tools_locations):
            for j in range(n_or_tools_locations):
                or_tools_distance_matrix[i, j] = or_tools_all_locations[i].distance_to(
                    or_tools_all_locations[j]
                )
        or_tools_demands_list = [0.0]
        for cust in valid_customers_for_or_tools:
            or_tools_demands_list.append(cust.demand)
        or_tools_demands = np.array(or_tools_demands_list)
        return (
            or_tools_distance_matrix,
            or_tools_demands,
            or_tools_all_locations, # This is the list of Location objects for OR-Tools
            or_tools_customer_indices_map,
        )

    def _prepare_routing_data(
        self,
        cluster_centroid: Location,
        customers: List[Location],
        robots: List[Robot], 
    ) -> Tuple[np.ndarray, np.ndarray, List[Location]]:
        """
        DEPRECATED
        """
        pass # Deprecated

    def _solve_with_routing_algorithm(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        robots: List[Robot],
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        if distance_matrix.shape[0] <= 1:
            return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}
        try:
            routes, total_distance = self.routing_algorithm.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                capacities=capacities,
                depot_index=0,
                max_vehicles=len(robots),
            )
            if routes and any(len(route) > 2 for route in routes):
                metrics = self._calculate_metrics(routes, distance_matrix, robots)
                return routes, metrics
        except Exception as e:
            print(f"Error solving robot routing problem: {str(e)}")
        return [], {"total_robot_distance": 0.0, "max_robot_time": 0.0, "total_time": 0.0}

    def _create_fallback_routes(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        robots: List[Robot],
    ) -> Tuple[List[List[int]], Dict[str, float], List[int]]: # Returns unassigned indices
        routes = []
        customer_original_indices = list(range(1, len(demands)))
        customers_to_assign_sorted = sorted(
            customer_original_indices,
            key=lambda cust_idx: demands[cust_idx],
            reverse=True,
        )
        unassigned_customer_indices_pool = customers_to_assign_sorted[:]
        for robot_idx, robot in enumerate(robots):
            if not unassigned_customer_indices_pool: break
            current_robot_route = [0]
            current_robot_load = 0.0
            customers_assigned_to_this_robot = []
            customers_still_unassigned_after_this_robot = []
            for customer_original_idx in unassigned_customer_indices_pool:
                customer_demand = demands[customer_original_idx]
                if current_robot_load + customer_demand <= robot.capacity:
                    current_robot_route.append(customer_original_idx)
                    current_robot_load += customer_demand
                    customers_assigned_to_this_robot.append(customer_original_idx)
                else:
                    customers_still_unassigned_after_this_robot.append(customer_original_idx)
            unassigned_customer_indices_pool = customers_still_unassigned_after_this_robot
            current_robot_route.append(0)
            if len(current_robot_route) > 2:
                routes.append(current_robot_route)
        metrics = self._calculate_metrics(routes, distance_matrix, robots)
        return routes, metrics, unassigned_customer_indices_pool

    def _calculate_metrics(
        self, routes: List[List[int]], distance_matrix: np.ndarray, robots: List[Robot]
    ) -> Dict[str, float]:
        total_robot_distance = 0.0
        robot_times = []
        for i, route in enumerate(routes):
            if not route or len(route) <= 2: continue
            robot_idx = min(i, len(robots) - 1)
            robot = robots[robot_idx]
            route_distance = sum(
                distance_matrix[route[j], route[j + 1]] for j in range(len(route) - 1)
            )
            total_robot_distance += route_distance
            travel_time = route_distance / robot.speed
            num_customers = len(route) - 2
            service_time = num_customers * 5.0 / 60.0
            launch_time = self.robot_launch_time / 60.0
            recovery_time = self.robot_recovery_time / 60.0
            recharge_time = 0.0
            battery_capacity_hours = robot.battery_capacity / 60.0
            if travel_time > battery_capacity_hours:
                recharge_time = (travel_time - battery_capacity_hours) * self.recharge_time_factor
            robot_time = travel_time + service_time + launch_time + recovery_time + recharge_time
            robot_times.append(robot_time)
        max_robot_time = max(robot_times) if robot_times else 0.0
        max_robot_time_minutes = max_robot_time * 60.0
        metrics = {
            "total_robot_distance": total_robot_distance,
            "max_robot_time": max_robot_time_minutes,
            "total_time": sum(robot_times),
            "individual_robot_times": robot_times,
            "num_active_robots": len(robot_times),
        }
        return metrics
