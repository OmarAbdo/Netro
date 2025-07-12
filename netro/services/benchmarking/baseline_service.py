# netro/services/benchmarking/baseline_service.py
from typing import List, Dict, Tuple, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck
from netro.core.interfaces.routing import RoutingAlgorithm


class BaselineTruckService:
    """
    Implementation of traditional truck-only delivery service.
    This is used as a benchmark to compare with the Netro approach.

    The baseline formula is:
    T(traditional) = N * (t(travel-customer) + t(service))

    where:
    - N: number of customers

    Based on:
    Solomon, M.M. (1987), "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints".
    """

    def __init__(
        self, routing_algorithm: RoutingAlgorithm, driver_hourly_cost: float = 15.0
    ):
        """
        Initialize the baseline truck service.

        Args:
            routing_algorithm: Algorithm to use for truck routing.
            driver_hourly_cost: Hourly cost for truck drivers in EUR.
        """
        self.routing_algorithm = routing_algorithm
        self.driver_hourly_cost = driver_hourly_cost

    def solve(
        self, depot: Location, customers: List[Location], trucks: List[Truck]
    ) -> Dict[str, Any]:
        """
        Solve the truck-only delivery problem.

        Args:
            depot: Depot location.
            customers: List of customer locations.
            trucks: List of available trucks.

        Returns:
            Dictionary with solution details and metrics.
        """
        # Create combined list of locations with depot as first location
        all_locations = [depot] + customers

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

        # Solve the routing problem
        routes, total_distance = self.routing_algorithm.solve(
            distance_matrix=distance_matrix,
            demands=demands,
            capacities=capacities,
            depot_index=0,
            max_vehicles=len(trucks),
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            routes, distance_matrix, trucks, all_locations
        )

        # Identify unserved customers
        # all_locations includes depot at index 0, then customers
        # routes contain indices relative to all_locations
        input_customer_indices = set(range(1, len(all_locations))) # 0 is depot
        served_customer_indices_in_routes = set()
        for route in routes:
            for node_idx in route[1:-1]: # Exclude depot at start/end of route
                served_customer_indices_in_routes.add(node_idx)

        unserved_customer_original_indices = input_customer_indices - served_customer_indices_in_routes
        
        # The 'customers' list passed to solve() are the original customer objects
        # We need to map unserved_customer_original_indices (which are indices into all_locations)
        # back to actual Location objects from the input 'customers' list.
        # An index 'k' in unserved_customer_original_indices corresponds to all_locations[k],
        # which in turn corresponds to customers[k-1] (because all_locations = [depot] + customers).
        unserved_customer_objects = [customers[idx - 1] for idx in unserved_customer_original_indices if (idx -1) < len(customers)]

        if unserved_customer_objects:
            print(f"[BaselineTruckService] Identified {len(unserved_customer_objects)} unserved customers from its input.")
            # for cust in unserved_customer_objects:
            #     print(f"  Unserved: ID {cust.id}")


        # Create and return the complete solution
        solution = {
            "routes": routes,
            "total_distance": total_distance,
            "total_time": metrics["total_time"],
            "total_cost": metrics["total_cost"],
            "total_emissions": metrics["total_emissions"],
            "metrics": metrics,
            "unserved_customers": unserved_customer_objects, # Add unserved customers
        }

        return solution

    def _calculate_metrics(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        trucks: List[Truck],
        locations: List[Location],
    ) -> Dict[str, float]:
        """
        Calculate detailed metrics for truck routes.

        Args:
            routes: List of truck routes.
            distance_matrix: Distance matrix.
            trucks: List of trucks.
            locations: List of locations (depot + customers).

        Returns:
            Dictionary with metrics.
        """
        total_distance = 0.0
        total_time = 0.0  # This will be the SUM of all route times
        total_cost = 0.0
        total_emissions = 0.0
        total_service_time = 0.0
        truck_utilization = []
        route_times = []

        for i, route in enumerate(routes):
            if not route or len(route) <= 2:  # Skip empty routes
                continue

            # Get the truck for this route
            truck_idx = min(i, len(trucks) - 1)
            truck = trucks[truck_idx]

            # Calculate route distance
            route_distance = 0.0
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]

            # Calculate metrics for this route
            travel_time = route_distance / truck.speed

            # Calculate service time for each customer
            service_time = 0.0
            for loc_idx in route[1:-1]:  # Skip depot
                loc = locations[loc_idx]
                if loc.service_time is not None:
                    service_time += loc.service_time
                else:
                    # Default service time if not specified
                    service_time += 5.0  # 5 minutes per customer

            # Convert service time from minutes to hours
            service_time_hours = service_time / 60.0

            # Calculate total route time
            route_time = travel_time + service_time_hours
            route_times.append(route_time)

            # Calculate other metrics
            route_cost = (
                truck.calculate_trip_cost(route_distance, route_time)
                if hasattr(truck, "calculate_trip_cost")
                else route_distance * 0.5
            )
            route_emissions = (
                truck.calculate_emissions(route_distance)
                if hasattr(truck, "calculate_emissions")
                else route_distance * 120.0
            )

            # Calculate truck utilization
            route_demand = sum(locations[loc_idx].demand for loc_idx in route[1:-1])
            utilization = route_demand / truck.capacity

            # Update totals - CORRECTED: total_time is sum of all route times
            total_distance += route_distance
            total_time += route_time  # Sum of all individual route times
            total_cost += route_cost
            total_emissions += route_emissions
            total_service_time += service_time_hours
            truck_utilization.append(utilization)

        # Compute averages
        avg_utilization = (
            sum(truck_utilization) / len(truck_utilization)
            if truck_utilization
            else 0.0
        )

        metrics = {
            "total_distance": total_distance,
            "total_time": total_time,  # Sum across all truck routes
            "travel_time": total_time - total_service_time,
            "service_time": total_service_time,
            "total_cost": total_cost,
            "total_emissions": total_emissions,
            "avg_truck_utilization": avg_utilization,
            "num_trucks_used": len([r for r in routes if len(r) > 2]),
            "route_times": route_times,  # Individual route times for reference
        }

        print(
            f"Baseline metrics: total_time={total_time:.2f}h, route_times={[f'{t:.1f}' for t in route_times]}"
        )

        return metrics
