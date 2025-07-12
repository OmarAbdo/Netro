# netro/services/routing/local_search.py
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import copy
import random
from netro.core.interfaces.routing import LocalSearch


class AdaptiveLocalSearch:
    """
    Implements an adaptive local search for vehicle routing problems using
    multiple operators and an adaptive operator selection mechanism.

    Based on:
    Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for
    dynamic vehicle routing problems", Computers and Electrical Engineering, 67, 596–607.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        improvement_threshold: float = 1e-4,
        max_iterations_without_improvement: int = 100,
    ):
        """
        Initialize the adaptive local search.

        Args:
            max_iterations: Maximum number of iterations.
            improvement_threshold: Minimum improvement to consider a solution better.
            max_iterations_without_improvement: Stop after this many iterations without improvement.
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.max_iterations_without_improvement = max_iterations_without_improvement

        # Initialize operators
        self.operators = {
            "two_opt": self._apply_2opt,
            "swap": self._apply_swap,
            "relocate": self._apply_relocate,
            "merge": self._apply_merge,
        }

        # Initialize operator weights (for adaptive selection)
        self.operator_weights = {op: 1.0 for op in self.operators}

    def improve(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
        **kwargs
    ) -> Tuple[List[List[int]], float]:
        """
        Improve routes using adaptive local search.

        Args:
            routes: List of routes, where each route is a list of location indices.
            distance_matrix: Matrix of distances between locations.
            demands: Array of demands for each location.
            capacities: Array of vehicle capacities.
            **kwargs: Additional parameters.

        Returns:
            A tuple containing:
            - Improved routes.
            - The new total distance.
        """
        current_solution = copy.deepcopy(routes)
        best_solution = copy.deepcopy(routes)
        best_distance = self._calculate_total_distance(best_solution, distance_matrix)

        iterations_without_improvement = 0

        for iteration in range(self.max_iterations):
            # Adaptive operator selection
            operator_name = self._select_operator()
            operator_func = self.operators[operator_name]

            # Apply selected operator
            new_solution = operator_func(
                current_solution, distance_matrix, demands, capacities
            )

            # Calculate new distance
            new_distance = self._calculate_total_distance(new_solution, distance_matrix)

            # Check if the new solution is better
            if new_distance < best_distance - self.improvement_threshold:
                best_solution = copy.deepcopy(new_solution)
                best_distance = new_distance
                current_solution = copy.deepcopy(new_solution)
                iterations_without_improvement = 0

                # Reward the successful operator
                self.operator_weights[operator_name] += 0.1
            else:
                iterations_without_improvement += 1

            # Normalize weights periodically
            if iteration % 50 == 0:
                self._normalize_weights()

            # Early termination if no improvement for a while
            if (
                iterations_without_improvement
                >= self.max_iterations_without_improvement
            ):
                break

        return best_solution, best_distance

    def _select_operator(self) -> str:
        """Select an operator based on weights using roulette wheel selection."""
        total_weight = sum(self.operator_weights.values())
        r = random.uniform(0, total_weight)
        cumulative = 0

        for op, weight in self.operator_weights.items():
            cumulative += weight
            if r <= cumulative:
                return op

        # Default fallback
        return list(self.operators.keys())[0]

    def _normalize_weights(self) -> None:
        """Normalize operator weights so they sum to the number of operators."""
        n_operators = len(self.operators)
        total = sum(self.operator_weights.values())

        if total > 0:
            for op in self.operator_weights:
                self.operator_weights[op] = (
                    self.operator_weights[op] * n_operators / total
                )

    def _calculate_total_distance(
        self, routes: List[List[int]], distance_matrix: np.ndarray
    ) -> float:
        """Calculate the total distance of all routes."""
        total = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                total += distance_matrix[route[i], route[i + 1]]
        return total

    def _apply_2opt(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
    ) -> List[List[int]]:
        """
        Apply 2-opt operator to improve routes by swapping edges within a route.

        Args:
            routes: Current routes.
            distance_matrix: Distance matrix.
            demands: Customer demands.
            capacities: Vehicle capacities.

        Returns:
            Improved routes.
        """
        new_routes = copy.deepcopy(routes)

        # Select a random route to improve
        if not new_routes:
            return new_routes

        route_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[route_idx]

        # Only apply 2-opt if route has enough nodes
        if len(route) <= 3:
            return new_routes

        # Try all possible 2-opt swaps and pick the best
        best_route = route.copy()
        best_distance = self._route_distance(route, distance_matrix)

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # Skip adjacent edges
                if j - i == 1:
                    continue

                # Create new route with a 2-opt swap
                new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                new_distance = self._route_distance(new_route, distance_matrix)

                if new_distance < best_distance - self.improvement_threshold:
                    best_route = new_route
                    best_distance = new_distance

        new_routes[route_idx] = best_route
        return new_routes

    def _apply_swap(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
    ) -> List[List[int]]:
        """
        Apply swap operator to exchange customers between two routes.

        Args:
            routes: Current routes.
            distance_matrix: Distance matrix.
            demands: Customer demands.
            capacities: Vehicle capacities.

        Returns:
            Improved routes.
        """
        if len(routes) <= 1:
            return routes

        new_routes = copy.deepcopy(routes)

        # Select two random routes
        route_indices = random.sample(range(len(new_routes)), 2)
        route1_idx, route2_idx = route_indices
        route1 = new_routes[route1_idx]
        route2 = new_routes[route2_idx]

        # Only proceed if both routes have customers
        if len(route1) <= 2 or len(route2) <= 2:
            return new_routes

        # Select a random customer from each route
        cust1_idx = random.randint(1, len(route1) - 2)
        cust2_idx = random.randint(1, len(route2) - 2)

        cust1 = route1[cust1_idx]
        cust2 = route2[cust2_idx]

        # Check capacity constraints
        route1_demand = sum(demands[i] for i in route1[1:-1])
        route2_demand = sum(demands[i] for i in route2[1:-1])

        new_route1_demand = route1_demand - demands[cust1] + demands[cust2]
        new_route2_demand = route2_demand - demands[cust2] + demands[cust1]

        if (
            new_route1_demand > capacities[route1_idx]
            or new_route2_demand > capacities[route2_idx]
        ):
            return new_routes

        # Perform the swap
        route1[cust1_idx], route2[cust2_idx] = route2[cust2_idx], route1[cust1_idx]

        # Calculate old and new distances
        old_distance = self._route_distance(
            routes[route1_idx], distance_matrix
        ) + self._route_distance(routes[route2_idx], distance_matrix)
        new_distance = self._route_distance(
            new_routes[route1_idx], distance_matrix
        ) + self._route_distance(new_routes[route2_idx], distance_matrix)

        # Only keep the change if it improves the solution
        if new_distance >= old_distance - self.improvement_threshold:
            return routes

        return new_routes

    def _apply_relocate(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
    ) -> List[List[int]]:
        """
        Apply relocate operator to move a customer from one route to another.

        Args:
            routes: Current routes.
            distance_matrix: Distance matrix.
            demands: Customer demands.
            capacities: Vehicle capacities.

        Returns:
            Improved routes.
        """
        if len(routes) <= 1:
            return routes

        new_routes = copy.deepcopy(routes)

        # Select a random source route
        source_idx = random.randint(0, len(new_routes) - 1)
        source_route = new_routes[source_idx]

        # Only proceed if source route has customers to relocate
        if len(source_route) <= 2:
            return new_routes

        # Select a random customer from source route
        cust_idx = random.randint(1, len(source_route) - 2)
        customer = source_route[cust_idx]

        # Select a random target route (different from source)
        target_indices = [i for i in range(len(new_routes)) if i != source_idx]
        if not target_indices:
            return new_routes

        target_idx = random.choice(target_indices)
        target_route = new_routes[target_idx]

        # Check capacity constraint for target route
        target_demand = sum(demands[i] for i in target_route[1:-1])
        new_target_demand = target_demand + demands[customer]

        if new_target_demand > capacities[target_idx]:
            return new_routes

        # Select a random position in target route to insert customer
        target_pos = random.randint(1, len(target_route) - 1)

        # Remove customer from source route
        source_route.pop(cust_idx)

        # Insert customer into target route
        target_route.insert(target_pos, customer)

        # Calculate old and new distances
        old_distance = self._route_distance(
            routes[source_idx], distance_matrix
        ) + self._route_distance(routes[target_idx], distance_matrix)
        new_distance = self._route_distance(
            new_routes[source_idx], distance_matrix
        ) + self._route_distance(new_routes[target_idx], distance_matrix)

        # Only keep the change if it improves the solution
        if new_distance >= old_distance - self.improvement_threshold:
            return routes

        return new_routes

    def _apply_merge(
        self,
        routes: List[List[int]],
        distance_matrix: np.ndarray,
        demands: np.ndarray,
        capacities: np.ndarray,
    ) -> List[List[int]]:
        """
        Apply merge operator to combine two routes if capacity allows.

        Args:
            routes: Current routes.
            distance_matrix: Distance matrix.
            demands: Customer demands.
            capacities: Vehicle capacities.

        Returns:
            Improved routes.
        """
        if len(routes) <= 1:
            return routes

        new_routes = copy.deepcopy(routes)

        # Select two random routes
        route_indices = random.sample(range(len(new_routes)), 2)
        route1_idx, route2_idx = route_indices
        route1 = new_routes[route1_idx]
        route2 = new_routes[route2_idx]

        # Check capacity constraint
        route1_demand = sum(demands[i] for i in route1[1:-1])
        route2_demand = sum(demands[i] for i in route2[1:-1])

        # Use the larger capacity of the two vehicles
        merged_capacity = max(capacities[route1_idx], capacities[route2_idx])

        if route1_demand + route2_demand > merged_capacity:
            return new_routes

        # Try different merge options
        # Option 1: route1 + route2
        merged_route1 = route1[:-1] + route2[1:]

        # Option 2: route2 + route1
        merged_route2 = route2[:-1] + route1[1:]

        # Calculate distances
        old_distance = self._route_distance(
            route1, distance_matrix
        ) + self._route_distance(route2, distance_matrix)
        merged1_distance = self._route_distance(merged_route1, distance_matrix)
        merged2_distance = self._route_distance(merged_route2, distance_matrix)

        # Choose the best merge option
        if merged1_distance <= merged2_distance:
            best_merged = merged_route1
            best_merged_distance = merged1_distance
        else:
            best_merged = merged_route2
            best_merged_distance = merged2_distance

        # Only merge if it improves the solution
        if best_merged_distance >= old_distance - self.improvement_threshold:
            return new_routes

        # Create the new solution with merged route
        result_routes = []
        for i, route in enumerate(new_routes):
            if i == route1_idx:
                result_routes.append(best_merged)
            elif i == route2_idx:
                continue  # Skip the second route that was merged
            else:
                result_routes.append(route)

        return result_routes

    def _route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate the distance of a single route."""
        if len(route) <= 1:
            return 0.0

        distance = 0.0
        for i in range(len(route) - 1):
            distance += distance_matrix[route[i], route[i + 1]]
        return distance
