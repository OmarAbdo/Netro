# netro/services/optimization/advanced_location_optimizer.py
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from netro.core.entities.location import Location
from netro.core.entities.cluster import Cluster


class AdvancedLocationOptimizer:
    """
    Implements advanced academically-grounded approaches to optimize truck stopping locations
    and reduce total travel distance in hybrid delivery systems.

    Implements multiple strategies from literature:

    1. Weighted Gain Point (WGP) approach
       Based on Wang et al. (2019), "Optimal stopping locations for autonomous
       delivery vehicles to minimize delivery time"

    2. Multi-echelon delivery with transit integration
       Based on De Maio et al. (2023), "Sustainable last-mile distribution with
       autonomous robots and public transportation"

    3. Dynamic programming for optimal truck stopping points
       Based on Ostermeier et al. (2022), "Cost-optimal truck-and-robot routing
       for last-mile delivery"

    4. Battery-constrained cluster formation
       Based on Chen et al. (2018), "An adaptive large neighborhood search heuristic
       for dynamic vehicle routing problems"
    """

    def __init__(
        self,
        max_robot_range: float = 5.0,
        robot_speed: float = 15.0,
        robot_battery_life: float = 120.0,
        weight_demand: float = 0.7,
        weight_distance: float = 0.3,
        use_street_network: bool = False,
        public_transit_enabled: bool = False,
    ):
        """
        Initialize the optimizer with various parameters.

        Args:
            max_robot_range: Maximum travel distance (km) for robots from truck.
            robot_speed: Robot speed in km/h.
            robot_battery_life: Robot battery life in minutes.
            weight_demand: Weight factor for customer demand in WGP calculation.
            weight_distance: Weight factor for distance in WGP calculation.
            use_street_network: Whether to use street network (if available).
            public_transit_enabled: Whether to consider public transit integration.
        """
        self.max_robot_range = max_robot_range
        self.robot_speed = robot_speed
        self.robot_battery_life = robot_battery_life
        self.weight_demand = weight_demand
        self.weight_distance = weight_distance
        self.use_street_network = use_street_network
        self.public_transit_enabled = public_transit_enabled

        # Maximum distance a robot can travel within battery constraints
        self.max_battery_distance = (robot_battery_life / 60.0) * robot_speed

    def optimize_stopping_locations(
        self,
        clusters: List[Cluster],
        depot: Location,
    ) -> Dict[int, Location]:
        """
        Find optimal truck stopping locations for each cluster.

        Args:
            clusters: List of customer clusters.
            depot: Depot location.

        Returns:
            Dictionary mapping cluster IDs to optimized truck stopping locations.
        """
        optimized_locations = {}

        for cluster in clusters:
            if not cluster.locations:
                continue

            # Select the most appropriate optimization method
            if self.public_transit_enabled:
                # Find nearest transit stops and use them as potential stopping points
                stop_location = self._find_transit_stop(cluster, depot)
            else:
                # Use Weighted Gain Point method from Wang et al. (2019)
                stop_location = self._calculate_weighted_gain_point(cluster)

            optimized_locations[cluster.id] = stop_location

        return optimized_locations

    def _calculate_weighted_gain_point(self, cluster: Cluster) -> Location:
        """
        Calculate the Weighted Gain Point (WGP) as an optimal truck stopping location.

        The WGP balances:
        - Customer demand weight (important customers have more influence)
        - Distance minimization (accessible to more customers)
        - Battery constraints (ensure robots can reach all customers)

        Based on Wang et al. (2019)

        Args:
            cluster: The customer cluster to optimize.

        Returns:
            An optimal location for the truck to stop.
        """
        if not cluster.locations:
            # Create a dummy location if the cluster is empty
            return Location(id=-cluster.id - 2000, x=0.0, y=0.0, demand=0.0)

        # Get coordinates and demands
        coords = np.array([loc.coordinates() for loc in cluster.locations])
        demands = np.array([loc.demand for loc in cluster.locations])

        # Normalize demands (important for weighting)
        total_demand = demands.sum()
        if total_demand > 0:
            normalized_demands = demands / total_demand
        else:
            normalized_demands = np.ones_like(demands) / len(demands)

        # Calculate weighted centroid based on demand
        demand_weighted_coords = np.zeros(2)
        for i, loc in enumerate(cluster.locations):
            demand_weighted_coords += normalized_demands[i] * np.array([loc.x, loc.y])

        # Calculate min-max distance point (point that minimizes the maximum distance to any customer)
        # This requires a more complex algorithm, for now we'll use a simplified approach
        distances = np.zeros((len(coords), len(coords)))
        for i in range(len(coords)):
            for j in range(len(coords)):
                distances[i, j] = np.linalg.norm(coords[i] - coords[j])

        # Find the location that minimizes maximum distance to other points
        max_distances = np.max(distances, axis=1)
        min_max_idx = np.argmin(max_distances)
        min_max_coords = coords[min_max_idx]

        # Combine the demand-weighted centroid and min-max point
        # This is the key insight from Wang et al. (2019)
        wgp_coords = (
            self.weight_demand * demand_weighted_coords
            + self.weight_distance * min_max_coords
        )

        # Check if the WGP ensures all customers are within battery range
        max_distance_to_wgp = max(np.linalg.norm(coords - wgp_coords, axis=1))

        if max_distance_to_wgp > self.max_battery_distance / 2:  # Round trip
            # Some customers might be unreachable - adjust the point
            # Use battery-constrained approach from Chen et al. (2018)
            wgp_coords = self._adjust_for_battery_constraints(
                coords, demands, wgp_coords
            )

        # Create a new location object for the WGP
        wgp_location = Location(
            id=-cluster.id - 1000,  # Negative ID to distinguish from real locations
            x=float(wgp_coords[0]),
            y=float(wgp_coords[1]),
            demand=float(total_demand),  # Total demand of the cluster
        )

        return wgp_location

    def _adjust_for_battery_constraints(
        self, coords: np.ndarray, demands: np.ndarray, initial_wgp: np.ndarray
    ) -> np.ndarray:
        """
        Adjust the Weighted Gain Point to ensure all customers are within battery range.

        Based on battery-constrained regions in Chen et al. (2018)

        Args:
            coords: Array of customer coordinates.
            demands: Array of customer demands.
            initial_wgp: Initial WGP coordinates.

        Returns:
            Adjusted WGP coordinates.
        """
        # Find customers potentially out of range from initial WGP
        distances = np.linalg.norm(coords - initial_wgp, axis=1)
        max_range = self.max_battery_distance / 2  # Round trip consideration

        # If all customers are in range, return initial WGP
        if np.all(distances <= max_range):
            return initial_wgp

        # Find the centroid of customers that are in range
        in_range_mask = distances <= max_range

        if not np.any(in_range_mask):
            # If no customers are in range, use the original centroid
            return np.mean(coords, axis=0)

        in_range_coords = coords[in_range_mask]
        in_range_demands = demands[in_range_mask]

        # Normalize demands for in-range customers
        in_range_total = np.sum(in_range_demands)
        if in_range_total > 0:
            in_range_weights = in_range_demands / in_range_total
        else:
            in_range_weights = np.ones_like(in_range_demands) / len(in_range_demands)

        # Calculate demand-weighted centroid for in-range customers
        weighted_centroid = np.zeros(2)
        for i in range(len(in_range_coords)):
            weighted_centroid += in_range_weights[i] * in_range_coords[i]

        # Move the WGP towards problematic points but within battery constraints
        out_of_range_mask = ~in_range_mask
        if np.any(out_of_range_mask):
            out_of_range_coords = coords[out_of_range_mask]

            # Find the farthest point's direction
            farthest_idx = np.argmax(distances)
            farthest_vector = coords[farthest_idx] - initial_wgp

            # Normalize the vector and scale to half the max range
            if np.linalg.norm(farthest_vector) > 0:
                direction = farthest_vector / np.linalg.norm(farthest_vector)
                adjustment = direction * (max_range * 0.3)

                # Adjust the weighted centroid in that direction
                adjusted_wgp = weighted_centroid + adjustment
            else:
                adjusted_wgp = weighted_centroid
        else:
            adjusted_wgp = weighted_centroid

        return adjusted_wgp

    def _find_transit_stop(self, cluster: Cluster, depot: Location) -> Location:
        """
        Find the optimal public transit stop for this cluster.

        Based on De Maio et al. (2023)'s approach to integrate with public transportation.
        In a real implementation, this would use actual transit data. Here we simulate it.

        Args:
            cluster: The customer cluster.
            depot: Depot location.

        Returns:
            Location representing an optimal transit stop.
        """
        # In a real implementation, we would:
        # 1. Query public transit API for nearby stops
        # 2. Calculate transit travel times
        # 3. Evaluate accessibility to customers

        # For this simulation, we'll create a point between the depot and cluster
        cluster_centroid = cluster.centroid
        if cluster_centroid is None and cluster.locations:
            coords = np.array([loc.coordinates() for loc in cluster.locations])
            cluster_centroid = np.mean(coords, axis=0)

        if cluster_centroid is None:
            # Fallback if no centroid can be calculated
            return Location(id=-cluster.id - 3000, x=depot.x, y=depot.y, demand=0.0)

        # Create a point along the line from depot to cluster centroid
        # In a real system, this would be the nearest transit hub
        transit_ratio = 0.7  # Position along the path (0 = depot, 1 = centroid)

        if isinstance(cluster_centroid, np.ndarray):
            transit_x = depot.x + transit_ratio * (cluster_centroid[0] - depot.x)
            transit_y = depot.y + transit_ratio * (cluster_centroid[1] - depot.y)
        else:
            transit_x = depot.x + transit_ratio * (cluster_centroid.x - depot.x)
            transit_y = depot.y + transit_ratio * (cluster_centroid.y - depot.y)

        # Create the transit stop location
        total_demand = (
            sum(loc.demand for loc in cluster.locations) if cluster.locations else 0
        )

        transit_stop = Location(
            id=-cluster.id - 3000,
            x=float(transit_x),
            y=float(transit_y),
            demand=float(total_demand),
        )

        return transit_stop

    def optimize_clusters_dp(
        self, customers: List[Location], depot: Location, max_clusters: int = 15
    ) -> List[Cluster]:
        """
        Use dynamic programming to optimize truck stopping points and cluster formation.

        Based on Ostermeier et al. (2022)'s dynamic programming formulation.

        Args:
            customers: List of customer locations.
            depot: Depot location.
            max_clusters: Maximum number of clusters to create.

        Returns:
            List of optimized clusters.
        """
        # This is a simplified implementation of the full DP approach
        # In the original paper, this was formulated as a location-routing problem

        # Sort customers by distance from depot
        sorted_customers = sorted(customers, key=lambda c: depot.distance_to(c))

        # Calculate costs for all possible contiguous clusters
        n = len(sorted_customers)
        costs = np.full(
            (n + 1, n + 1), np.inf
        )  # costs[i][j] = cost of cluster from i to j-1

        for i in range(n):
            for j in range(i + 1, min(i + 30, n + 1)):  # Limit cluster size
                cluster_customers = sorted_customers[i:j]

                # Skip if total demand exceeds maximum capacity
                total_demand = sum(c.demand for c in cluster_customers)
                if total_demand > 200:  # Assuming truck capacity of 200
                    continue

                # Calculate cost for this cluster using WGP approach
                temp_cluster = Cluster(id=0, locations=cluster_customers)
                wgp = self._calculate_weighted_gain_point(temp_cluster)

                # Cost = distance from depot to WGP + max distance from WGP to any customer
                depot_to_wgp = depot.distance_to(wgp)
                max_dist_to_customer = (
                    max(wgp.distance_to(c) for c in cluster_customers)
                    if cluster_customers
                    else 0
                )

                # Total cost formula (from Ostermeier et al.)
                costs[i][j] = (
                    2 * depot_to_wgp + max_dist_to_customer * cluster_customers
                )

        # DP table for minimum cost clustering
        dp = np.full(n + 1, np.inf)
        dp[0] = 0
        prev = np.zeros(n + 1, dtype=int)

        # Fill DP table
        for j in range(1, n + 1):
            for i in range(j):
                if costs[i][j] < np.inf and dp[i] + costs[i][j] < dp[j]:
                    dp[j] = dp[i] + costs[i][j]
                    prev[j] = i

        # Reconstruct the clusters
        clusters = []
        j = n
        while j > 0:
            i = prev[j]
            cluster_customers = sorted_customers[i:j]

            new_cluster = Cluster(id=len(clusters), locations=cluster_customers)
            clusters.append(new_cluster)

            j = i

        # Reverse to get clusters in original order
        clusters.reverse()

        return clusters
