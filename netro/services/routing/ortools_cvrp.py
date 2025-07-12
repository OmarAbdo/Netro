# netro/services/routing/ortools_cvrp.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from netro.core.interfaces.routing import RoutingAlgorithm


class ORToolsCVRP:
    """
    Capacitated Vehicle Routing Problem solver using Google OR-Tools.

    Based on:
    Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
    """

    def __init__(
        self,
        first_solution_strategy: str = "PATH_CHEAPEST_ARC",
        local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
        time_limit_seconds: int = 30,
    ):
        """
        Initialize OR-Tools CVRP solver.

        Args:
            first_solution_strategy: Strategy for finding the first solution.
            local_search_metaheuristic: Metaheuristic to use for local search.
            time_limit_seconds: Time limit for the solver in seconds.
        """
        self.first_solution_strategy = getattr(
            routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
        )
        self.local_search_metaheuristic = getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
        )
        self.time_limit_seconds = time_limit_seconds

    def solve(
        self,
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """
        Solve a capacitated vehicle routing problem using OR-Tools.

        Args:
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location (index 0 is typically depot).
            capacities: Array of vehicle capacities.
            **kwargs: Additional parameters:
                - depot_index: Index of the depot (default: 0)
                - max_vehicles: Maximum number of vehicles to use (default: len(capacities))

        Returns:
            A tuple containing:
            - A list of routes, where each route is a list of location indices.
            - The total distance of all routes.
        """
        # Process kwargs
        depot_index = kwargs.get("depot_index", 0)
        max_vehicles = kwargs.get("max_vehicles", len(capacities))

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix), max_vehicles, depot_index
        )

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Prevent node dropping with high penalty
        penalty = 1000000  # Arbitrarily large penalty value
        for node in range(1, len(demands)):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Define distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Define demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return int(demands[from_node])

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # Add capacity dimension
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            capacities.astype(int).tolist(),  # vehicle capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.first_solution_strategy
        search_parameters.local_search_metaheuristic = self.local_search_metaheuristic
        search_parameters.time_limit.seconds = self.time_limit_seconds

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if not solution: # This is the primary check if a solution object was returned at all
            print(f"[DEBUG] ORToolsCVRP.solve: routing.SolveWithParameters returned no solution object (solver status: {routing.status()})")
            return [], 0.0
        
        # ---- MODIFIED DEBUG PRINT for status 4 (if solution object exists) ----
        if routing.status() == 4: # ROUTING_SUCCESS
            current_routes_for_debug = []
            for i in range(max_vehicles):
                vehicle_route = []
                index = routing.Start(i)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    vehicle_route.append(node_index)
                    index = solution.Value(routing.NextVar(index))
                node_index = manager.IndexToNode(index) # Add the end node (depot)
                vehicle_route.append(node_index)
                # Only add if it's a non-trivial route (visits at least one customer)
                if len(vehicle_route) > 2 or (len(vehicle_route) == 2 and vehicle_route[0] != vehicle_route[1]): # Check for depot-only routes
                     current_routes_for_debug.append(vehicle_route)
            print(f"[DEBUG] ORToolsCVRP.solve: Solver status 4 (ROUTING_SUCCESS). Raw routes extracted by debug logic: {current_routes_for_debug}")
        elif routing.status() != 1: # If not ROUTING_SUCCESS (status 1 is also success, but 4 is more common for solved) and not 4
             print(f"[DEBUG] ORToolsCVRP.solve: Solution object exists, but status is not ideal: {routing.status()}")
        # ---- END MODIFIED DEBUG PRINT ----

        # Extract solution (original logic)
        routes = []
        total_distance = 0

        for vehicle_id in range(max_vehicles):
            index = routing.Start(vehicle_id)
            route = [manager.IndexToNode(index)]
            route_distance = 0

            while not routing.IsEnd(index):
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            if len(route) > 2:  # Only include non-empty routes
                routes.append(route)
                total_distance += route_distance

        return routes, float(total_distance)
