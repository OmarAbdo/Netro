# netro/infrastructure/visualization/solution_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from netro.core.entities.location import Location
from netro.core.entities.cluster import Cluster


class SolutionVisualizer:
    """
    Visualization tools for displaying clustering and routing solutions.
    """

    @staticmethod
    def plot_clusters(
        clusters: List[Cluster],
        depot: Location,
        centroids: Dict[int, Location] = None,
        title: str = "Customer Clusters",
    ) -> plt.Figure:
        """
        Plot customer clusters with centroids and depot.

        Args:
            clusters: List of Cluster objects.
            depot: Depot location.
            centroids: Optional dictionary of centroid locations by cluster ID.
            title: Plot title.

        Returns:
            Matplotlib figure with the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a different colormap with more distinguishable colors
        cmap = plt.cm.get_cmap("tab20", len(clusters))

        # Plot each cluster
        for i, cluster in enumerate(clusters):
            # Extract coordinates
            x_coords = [loc.x for loc in cluster.locations]
            y_coords = [loc.y for loc in cluster.locations]

            # Plot cluster points
            ax.scatter(
                x_coords,
                y_coords,
                c=[cmap(i)],
                label=f"Cluster {cluster.id}",
                alpha=0.7,
            )

            # Plot cluster centroid if available in the cluster
            if cluster.centroid is not None:
                ax.scatter(
                    cluster.centroid[0],
                    cluster.centroid[1],
                    marker="x",
                    color=cmap(i),
                    s=100,
                )

        # Plot additional centroids if provided
        if centroids:
            for cluster_id, centroid in centroids.items():
                ax.scatter(
                    centroid.x,
                    centroid.y,
                    marker="*",
                    color="black",
                    s=150,
                    label=(
                        "Centroids" if cluster_id == list(centroids.keys())[0] else ""
                    ),
                )

        # Plot depot
        ax.scatter(depot.x, depot.y, c="red", marker="s", s=100, label="Depot")

        # Set labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)

        # Add legend (only show a reasonable number of entries)
        if len(clusters) > 10:
            ax.legend(loc="best", ncol=2)
        else:
            ax.legend(loc="best")

        ax.grid(True, alpha=0.3)

        return fig

    @staticmethod
    def plot_truck_routes(
        routes: List[List[int]], locations: List[Location], title: str = "Truck Routes"
    ) -> plt.Figure:
        """
        Plot truck routes.

        Args:
            routes: List of routes, where each route is a list of location indices.
            locations: List of locations (depot at index 0).
            title: Plot title.

        Returns:
            Matplotlib figure with the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract coordinates for all locations
        x_coords = [loc.x for loc in locations]
        y_coords = [loc.y for loc in locations]

        # Plot all locations
        ax.scatter(x_coords[1:], y_coords[1:], c="blue", alpha=0.5, label="Customers")

        # Plot depot
        ax.scatter(x_coords[0], y_coords[0], c="red", marker="s", s=100, label="Depot")

        # Plot each route with a different color
        cmap = plt.cm.get_cmap("tab10", len(routes))

        for i, route in enumerate(routes):
            if len(route) <= 1:
                continue

            route_x = [locations[idx].x for idx in route]
            route_y = [locations[idx].y for idx in route]

            ax.plot(
                route_x,
                route_y,
                "o-",
                color=cmap(i),
                linewidth=2,
                markersize=5,
                label=f"Route {i+1}",
            )

        # Set labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)

        # Add legend
        if len(routes) > 10:
            ax.legend(loc="best", ncol=2)
        else:
            ax.legend(loc="best")

        ax.grid(True, alpha=0.3)

        return fig

    @staticmethod
    def plot_netro_solution(
        depot: Location,
        clusters: List[Cluster],
        truck_routes: List[List[int]],
        cluster_routes: Dict[int, List[List[int]]],
        centroids: Dict[int, Location],
        title: str = "Netro Hybrid Solution",
    ) -> plt.Figure:
        """
        Plot the complete Netro solution with truck routes and robot routes.

        Args:
            depot: Depot location.
            clusters: List of Cluster objects.
            truck_routes: List of truck routes (to cluster centroids).
            cluster_routes: Dictionary mapping cluster IDs to lists of robot routes.
            centroids: Dictionary mapping cluster IDs to centroid locations.
            title: Plot title.

        Returns:
            Matplotlib figure with the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use different colormaps for clusters and routes
        cluster_cmap = plt.cm.get_cmap("tab20", len(clusters))
        truck_cmap = plt.cm.get_cmap("Dark2", len(truck_routes))

        # Plot clusters
        for i, cluster in enumerate(clusters):
            # Extract coordinates
            x_coords = [loc.x for loc in cluster.locations]
            y_coords = [loc.y for loc in cluster.locations]

            # Plot cluster points
            ax.scatter(
                x_coords,
                y_coords,
                c=[cluster_cmap(i)],
                label=f"Cluster {cluster.id}",
                alpha=0.6,
                s=30,
            )

        # Plot centroids
        for cluster_id, centroid in centroids.items():
            ax.scatter(centroid.x, centroid.y, marker="*", color="black", s=150)

        # Plot depot
        ax.scatter(depot.x, depot.y, c="red", marker="s", s=150, label="Depot")

        # Plot truck routes to centroids
        for i, route in enumerate(truck_routes):
            if len(route) <= 1:
                continue

            # Translate route indices to actual locations
            route_locations = (
                [depot] + [centroids.get(loc_idx) for loc_idx in route[1:-1]] + [depot]
            )
            route_x = [loc.x for loc in route_locations if loc is not None]
            route_y = [loc.y for loc in route_locations if loc is not None]

            ax.plot(
                route_x,
                route_y,
                "o-",
                color=truck_cmap(i),
                linewidth=3,
                markersize=8,
                label=f"Truck Route {i+1}",
            )

        # Plot robot routes within clusters
        robot_cmap = plt.cm.get_cmap("Paired", 12)

        for cluster_id, robot_routes in cluster_routes.items():
            if cluster_id not in centroids:
                continue

            centroid = centroids[cluster_id]
            cluster = next((c for c in clusters if c.id == cluster_id), None)

            if cluster is None:
                continue

            # Get all locations in the cluster
            cluster_locations = [centroid] + cluster.locations

            for j, route in enumerate(robot_routes):
                if len(route) <= 1:
                    continue

                # Map route indices to actual cluster locations
                route_locations = [
                    cluster_locations[idx]
                    for idx in route
                    if idx < len(cluster_locations)
                ]
                route_x = [loc.x for loc in route_locations]
                route_y = [loc.y for loc in route_locations]

                # Use a different linestyle for robot routes
                ax.plot(
                    route_x,
                    route_y,
                    "--",
                    color=robot_cmap(j % 12),
                    linewidth=1,
                    alpha=0.7,
                )

        # Set labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)

        # Create a custom legend
        ax.legend(loc="best", ncol=2)

        ax.grid(True, alpha=0.3)

        return fig
