# netro/services/routing/netro_routing_service.py
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot
from netro.core.entities.cluster import Cluster
from netro.core.interfaces.routing import RoutingAlgorithm


class NetroRoutingService:
    """
    High-level service for the complete Netro routing solution.
    Orchestrates the truck routing to clusters and robot delivery within clusters.

    This service implements the Netro formula:
    T(Netro) = K * (t(travel-cluster) + t(unloading)) + [M/R * (t(travel-robot-customer) + t(robot-service)) + t(recovery)]

    where:
    - K: number of clusters
    - M: customers per cluster
    - R: robots per cluster
    """

    def __init__(
        self,
        truck_routing_algorithm: RoutingAlgorithm,
        robot_routing_service: Any,  # Using Any to avoid circular import
        robot_unloading_time: float = 5.0,
    ):
        """
        Initialize the Netro routing service.

        Args:
            truck_routing_algorithm: Algorithm for routing trucks between clusters.
            robot_routing_service: Service for routing robots within clusters.
            robot_unloading_time: Time to unload all robots from a truck in minutes.
        """
        self.truck_routing_algorithm = truck_routing_algorithm
        self.robot_routing_service = robot_routing_service
        self.robot_unloading_time = robot_unloading_time

    def solve(
        self,
        depot: Location,
        clusters: List[Cluster],
        centroids: Dict[int, Location],
        trucks: List[Truck],
        robots_per_truck: List[List[Robot]],
    ) -> Dict[str, Any]:
        """
        Solve the complete Netro routing problem.

        Args:
            depot: Depot location.
            clusters: List of customer clusters.
            centroids: Dictionary mapping cluster ID to centroid location.
            trucks: List of available trucks.
            robots_per_truck: List of robots available for each truck.

        Returns:
            Dictionary with solution details and metrics.
        """
        # First, route trucks between cluster centroids
        truck_routes, truck_metrics = self._route_trucks(depot, centroids, trucks)
        
        print(f"DEBUG: Generated {len(truck_routes)} truck routes")
        for i, route in enumerate(truck_routes):
            print(f"DEBUG: Truck route {i}: {route}")

        # Then, for each truck route, route robots within each cluster
        cluster_routes = {}
        cluster_metrics = {}

        # Create a mapping from location_idx to cluster_id for easier lookup
        location_to_cluster = {}
        for cluster in clusters:
            for centroid_idx, centroid in centroids.items():
                # Try different ways to match centroids to clusters
                if centroid.id < 0:  # If using negative IDs for centroids
                    potential_cluster_id = -centroid.id - 1000
                    if potential_cluster_id == cluster.id:
                        location_to_cluster[centroid_idx] = cluster.id
                else:  # Try direct matching
                    if centroid_idx == cluster.id:
                        location_to_cluster[centroid_idx] = cluster.id
        
        print(f"DEBUG: Location to cluster mapping: {location_to_cluster}")
        print(f"DEBUG: Available clusters: {[c.id for c in clusters]}")
        print(f"DEBUG: Centroid IDs: {list(centroids.keys())}")
        print(f"DEBUG: Centroid location IDs: {[c.id for c in centroids.values()]}")

        for truck_idx, truck_route in enumerate(truck_routes):
            truck_route_metrics = {
                "cluster_times": [],
                "total_robot_distance": 0.0,
                "total_time": 0.0,
            }

            truck = trucks[truck_idx]
            truck_robots = robots_per_truck[truck_idx]
            
            print(f"DEBUG: Processing truck {truck_idx} with {len(truck_robots)} robots")

            for location_idx in truck_route:
                # Skip depot
                if location_idx == 0:
                    continue
                
                print(f"DEBUG: Processing location_idx {location_idx}")
                
                # Try to find the cluster ID using our mapping
                cluster_id = None
                if location_idx in location_to_cluster:
                    cluster_id = location_to_cluster[location_idx]
                else:
                    # Fallback to the original method
                    try:
                        cluster_id = -centroids[location_idx].id - 1000
                        print(f"DEBUG: Using fallback method, calculated cluster_id: {cluster_id}")
                    except Exception as e:
                        print(f"DEBUG: Error calculating cluster_id: {str(e)}")
                
                # Find the cluster object
                cluster_obj = None
                for cluster in clusters:
                    if cluster.id == cluster_id:
                        cluster_obj = cluster
                        break
                
                if cluster_obj is None:
                    print(f"DEBUG: No cluster found for ID {cluster_id}, skipping")
                    continue
                
                print(f"DEBUG: Found cluster {cluster_id} with {len(cluster_obj.locations)} locations")

                cluster = cluster_obj
                centroid = centroids[location_idx]

                # Route robots within this cluster
                try:
                    robot_routes, robot_metrics = self.robot_routing_service.route_robots(
                        cluster_centroid=centroid,
                        customers=cluster.locations,
                        robots=truck_robots,
                        truck=truck,
                    )
                    
                    print(f"DEBUG: Generated {len(robot_routes)} robot routes for cluster {cluster_id}")
                    for i, route in enumerate(robot_routes):
                        print(f"DEBUG: Robot route {i}: {route}")
                    print(f"DEBUG: Robot metrics: {robot_metrics}")

                    # Store the robot routes for this cluster
                    cluster_routes[cluster_id] = robot_routes

                    # Update metrics
                    cluster_time = (
                        self.robot_unloading_time + robot_metrics["max_robot_time"]
                    )
                    truck_route_metrics["cluster_times"].append(cluster_time)
                    truck_route_metrics["total_robot_distance"] += robot_metrics[
                        "total_robot_distance"
                    ]
                    truck_route_metrics["total_time"] += cluster_time
                except Exception as e:
                    print(f"DEBUG: Error routing robots: {str(e)}")

            # Store metrics for this truck route
            cluster_metrics[truck_idx] = truck_route_metrics

        # Calculate overall metrics
        total_time = truck_metrics["total_time"]
        for truck_idx, metrics in cluster_metrics.items():
            total_time += metrics["total_time"]

        # Create and return the complete solution
        solution = {
            "truck_routes": truck_routes,
            "cluster_routes": cluster_routes,
            "truck_metrics": truck_metrics,
            "cluster_metrics": cluster_metrics,
            "total_time": total_time,
            "total_truck_distance": truck_metrics["total_distance"],
            "total_robot_distance": sum(
                m["total_robot_distance"] for m in cluster_metrics.values()
            ),
        }

        return solution

    def _route_trucks(
        self, depot: Location, centroids: Dict[int, Location], trucks: List[Truck]
    ) -> Tuple[List[List[int]], Dict[str, float]]:
        """
        Route trucks between cluster centroids.

        Args:
            depot: Depot location.
            centroids: Dictionary mapping cluster ID to centroid location.
            trucks: List of available trucks.

        Returns:
            A tuple containing:
            - List of truck routes.
            - Dictionary with metrics.
        """
        # Create list of locations with depot as first location
        centroid_locations = list(centroids.values())
        all_locations = [depot] + centroid_locations
        
        print(f"DEBUG: Routing trucks between depot and {len(centroid_locations)} centroids")
        print(f"DEBUG: Depot: ({depot.x}, {depot.y})")
        
        # Compute distance matrix
        n_locations = len(all_locations)
        distance_matrix = np.zeros((n_locations, n_locations))

        for i in range(n_locations):
            for j in range(n_locations):
                distance_matrix[i, j] = all_locations[i].distance_to(all_locations[j])

        # Extract demands
        demands = np.array([loc.demand for loc in all_locations])
        print(f"DEBUG: Centroid demands: {demands}")

        # Truck capacities
        capacities = np.array([truck.capacity for truck in trucks])
        print(f"DEBUG: Truck capacities: {capacities}")
        
        # Check if we have any centroids to route to
        if len(centroid_locations) == 0:
            print("DEBUG: No centroids to route to, returning empty routes")
            return [], {"total_distance": 0.0, "total_time": 0.0}
            
        # Check if we have any trucks
        if len(trucks) == 0:
            print("DEBUG: No trucks available, returning empty routes")
            return [], {"total_distance": 0.0, "total_time": 0.0}

        # First, try to use the routing algorithm
        try:
            routes, total_distance = self.truck_routing_algorithm.solve(
                distance_matrix=distance_matrix,
                demands=demands,
                capacities=capacities,
                depot_index=0,
                max_vehicles=len(trucks),
            )
            
            print(f"DEBUG: Truck routing algorithm returned {len(routes)} routes with total distance {total_distance}")
            
            # If routes were generated, use them
            if routes:
                # Calculate truck metrics
                total_time = 0.0
                for i, route in enumerate(routes):
                    if not route:
                        continue

                    # Get the truck for this route
                    truck = trucks[min(i, len(trucks) - 1)]

                    # Calculate route distance
                    route_distance = 0.0
                    for j in range(len(route) - 1):
                        route_distance += distance_matrix[route[j], route[j + 1]]

                    # Calculate route time
                    route_time = route_distance / truck.speed
                    total_time += route_time
                    
                    print(f"DEBUG: Truck route {i} distance: {route_distance}, time: {route_time}")

                metrics = {"total_distance": total_distance, "total_time": total_time}
                print(f"DEBUG: Total truck metrics: {metrics}")
                
                return routes, metrics
        except Exception as e:
            print(f"DEBUG: Error solving truck routing problem: {str(e)}")
            # Fall through to the capacity-aware bin packing algorithm
        
        # If we get here, the routing algorithm didn't work or didn't generate routes
        # Use a capacity-aware bin packing algorithm to create routes
        print("DEBUG: Using capacity-aware bin packing to create truck routes")
        
        # Create a list of centroids with their demands and indices
        centroid_info = []
        for i in range(1, n_locations):  # Skip depot (index 0)
            centroid_info.append({
                "index": i,
                "demand": demands[i],
                "location": all_locations[i]
            })
        
        # Sort centroids by demand (descending) for better bin packing
        centroid_info.sort(key=lambda x: x["demand"], reverse=True)
        
        # Create routes using a bin packing algorithm
        routes = []
        total_distance = 0.0
        
        # Keep track of which centroids have been assigned to a route
        assigned_centroids = set()
        
        # First-fit decreasing bin packing algorithm
        for truck_idx, truck in enumerate(trucks):
            if len(assigned_centroids) == len(centroid_info):
                # All centroids have been assigned
                break
                
            route = [0]  # Start at depot
            remaining_capacity = truck.capacity
            
            # Try to add centroids to this truck's route
            for centroid in centroid_info:
                if centroid["index"] in assigned_centroids:
                    # This centroid has already been assigned to a route
                    continue
                    
                if centroid["demand"] <= remaining_capacity:
                    # This centroid can be added to the route
                    route.append(centroid["index"])
                    assigned_centroids.add(centroid["index"])
                    remaining_capacity -= centroid["demand"]
            
            if len(route) > 1:  # Only add route if it visits at least one centroid
                route.append(0)  # Return to depot
                routes.append(route)
                
                # Calculate route distance
                route_distance = 0.0
                for j in range(len(route) - 1):
                    route_distance += distance_matrix[route[j], route[j + 1]]
                total_distance += route_distance
                
                print(f"DEBUG: Created truck route {truck_idx}: {route} with distance {route_distance}")
        
        # If there are still unassigned centroids, create additional routes
        # This might happen if we have more centroids than trucks, or if some centroids have high demands
        remaining_centroids = [c for c in centroid_info if c["index"] not in assigned_centroids]
        
        if remaining_centroids and len(routes) < len(trucks):
            print(f"DEBUG: {len(remaining_centroids)} centroids still unassigned, creating additional routes")
            
            # Reuse trucks in round-robin fashion, but only up to the number of available trucks
            truck_idx = len(routes)  # Start with the next available truck
            
            while remaining_centroids and len(routes) < len(trucks):
                # Get the next truck
                truck = trucks[truck_idx]
                
                route = [0]  # Start at depot
                remaining_capacity = truck.capacity
                
                # Try to add centroids to this truck's route
                i = 0
                while i < len(remaining_centroids):
                    centroid = remaining_centroids[i]
                    
                    if centroid["demand"] <= remaining_capacity:
                        # This centroid can be added to the route
                        route.append(centroid["index"])
                        remaining_capacity -= centroid["demand"]
                        remaining_centroids.pop(i)  # Remove from remaining centroids
                    else:
                        # Try the next centroid
                        i += 1
                
                if len(route) > 1:  # Only add route if it visits at least one centroid
                    route.append(0)  # Return to depot
                    routes.append(route)
                    
                    # Calculate route distance
                    route_distance = 0.0
                    for j in range(len(route) - 1):
                        route_distance += distance_matrix[route[j], route[j + 1]]
                    total_distance += route_distance
                    
                    print(f"DEBUG: Created additional truck route: {route} with distance {route_distance}")
                
                # Move to the next truck
                truck_idx += 1
                
                # If we've used all available trucks, stop
                if truck_idx >= len(trucks):
                    break
            
            # If there are still unassigned centroids, log a warning
            if remaining_centroids:
                print(f"DEBUG: Could not assign {len(remaining_centroids)} centroids to any truck due to capacity constraints or truck availability")
        
        # Calculate truck metrics
        total_time = 0.0
        for i, route in enumerate(routes):
            if not route:
                continue

            # Get the truck for this route
            truck = trucks[min(i, len(trucks) - 1)]

            # Calculate route distance
            route_distance = 0.0
            for j in range(len(route) - 1):
                route_distance += distance_matrix[route[j], route[j + 1]]

            # Calculate route time
            route_time = route_distance / truck.speed
            total_time += route_time
            
            print(f"DEBUG: Truck route {i} distance: {route_distance}, time: {route_time}")

        metrics = {"total_distance": total_distance, "total_time": total_time}
        print(f"DEBUG: Total truck metrics: {metrics}")

        return routes, metrics
