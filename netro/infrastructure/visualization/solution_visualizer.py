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
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap("tab20", len(clusters) if clusters else 1)

        for i, cluster in enumerate(clusters):
            x_coords = [loc.x for loc in cluster.locations]
            y_coords = [loc.y for loc in cluster.locations]
            ax.scatter(
                x_coords,
                y_coords,
                c=[cmap(i)],
                label=f"Cluster {cluster.id}",
                alpha=0.7,
            )
            if cluster.centroid is not None:
                ax.scatter(
                    cluster.centroid[0],
                    cluster.centroid[1],
                    marker="x",
                    color=cmap(i),
                    s=100,
                )

        if centroids:
            for i, (cluster_id, centroid) in enumerate(centroids.items()):
                ax.scatter(
                    centroid.x,
                    centroid.y,
                    marker="*",
                    color="black",
                    s=150,
                    label="Centroids" if i == 0 else "",
                )

        ax.scatter(depot.x, depot.y, c="red", marker="s", s=100, label="Depot")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels for centroids if any
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", ncol=2 if len(by_label) > 5 else 1)
        ax.grid(True, alpha=0.3)
        return fig

    @staticmethod
    def plot_truck_routes(
        routes: List[List[int]], 
        locations: List[Location], 
        title: str = "Truck Routes",
        route_color_map: Optional[Any] = None,
        line_style: str = 'o-',
        line_width: int = 2,
        marker_size: int = 5,
        show_labels: bool = True,
        ax: Optional[plt.Axes] = None # Allow passing an existing Axes object
    ) -> plt.Figure:
        """
        Plot truck routes.
        """
        if ax is None:
            fig, ax_new = plt.subplots(figsize=(10, 8))
            ax = ax_new # Use the new Axes
        else:
            fig = ax.get_figure() # Get figure from existing Axes

        x_coords = [loc.x for loc in locations]
        y_coords = [loc.y for loc in locations]

        if show_labels:
            ax.scatter(x_coords[1:], y_coords[1:], c="blue", alpha=0.5, label="Customers")
            ax.scatter(x_coords[0], y_coords[0], c="red", marker="s", s=100, label="Depot")

        num_routes = len(routes)
        cmap = route_color_map if route_color_map else plt.cm.get_cmap("tab10", num_routes if num_routes > 0 else 1)

        for i, route_indices in enumerate(routes):
            if len(route_indices) <= 1:
                continue
            
            route_actual_locations = [locations[idx] for idx in route_indices]
            route_x = [loc.x for loc in route_actual_locations]
            route_y = [loc.y for loc in route_actual_locations]

            ax.plot(
                route_x,
                route_y,
                line_style,
                color=cmap(i % cmap.N if isinstance(cmap, plt.cm.colors.ListedColormap) or isinstance(cmap, plt.cm.colors.LinearSegmentedColormap) else i), # Handle different cmap types
                linewidth=line_width,
                markersize=marker_size,
                label=f"Route {i+1}" if show_labels else None,
            )
        
        if show_labels:
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.set_title(title)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="best", ncol=2 if len(by_label) > 5 else 1)
        
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
        last_resort_truck_routes: Optional[List[List[int]]] = None,
        all_locations_list: Optional[List[Location]] = None 
    ) -> plt.Figure:
        """
        Plot the complete Netro solution.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        cluster_cmap = plt.cm.get_cmap("tab20", len(clusters) if clusters else 1)
        truck_cmap = plt.cm.get_cmap("Dark2", len(truck_routes) if truck_routes else 1)
        robot_cmap = plt.cm.get_cmap("Paired", 12)
        
        num_last_resort_routes = len(last_resort_truck_routes) if last_resort_truck_routes else 0
        last_resort_cmap = plt.cm.get_cmap("cool", num_last_resort_routes if num_last_resort_routes > 0 else 1)


        all_customer_x = []
        all_customer_y = []
        for cluster in clusters:
            all_customer_x.extend([loc.x for loc in cluster.locations])
            all_customer_y.extend([loc.y for loc in cluster.locations])
        if all_customer_x: 
             ax.scatter(all_customer_x, all_customer_y, c="gray", alpha=0.3, label="Customers (by Robots/Fallback)", s=20)

        for i, cluster in enumerate(clusters):
            if cluster.id in centroids: 
                 ax.scatter(centroids[cluster.id].x, centroids[cluster.id].y, marker="*", color=cluster_cmap(i % cluster_cmap.N), s=200, label=f"Centroid C{cluster.id}", edgecolors='black')

        ax.scatter(depot.x, depot.y, c="red", marker="s", s=150, label="Depot", zorder=5)

        for i, route_centroid_ids in enumerate(truck_routes):
            if not route_centroid_ids or len(route_centroid_ids) < 3: 
                continue
            
            route_points_x = [depot.x]
            route_points_y = [depot.y]
            for centroid_key_idx in route_centroid_ids[1:-1]: 
                centroid_loc = centroids.get(centroid_key_idx)
                if centroid_loc:
                    route_points_x.append(centroid_loc.x)
                    route_points_y.append(centroid_loc.y)
            route_points_x.append(depot.x)
            route_points_y.append(depot.y)
            
            ax.plot(route_points_x, route_points_y, "o-", color=truck_cmap(i % truck_cmap.N), linewidth=2.5, markersize=7, label=f"Truck Route to Centroids {i+1}", zorder=3)

        for cluster_id, robot_routes_in_cluster in cluster_routes.items():
            if cluster_id not in centroids: continue
            centroid_loc = centroids[cluster_id]
            
            current_cluster = next((c for c in clusters if c.id == cluster_id), None)
            if not current_cluster: continue
            
            robot_route_locations_base = [centroid_loc] + current_cluster.locations

            for j, robot_route_indices in enumerate(robot_routes_in_cluster):
                if len(robot_route_indices) <=1: continue
                
                route_points_x = [robot_route_locations_base[idx].x for idx in robot_route_indices if idx < len(robot_route_locations_base)]
                route_points_y = [robot_route_locations_base[idx].y for idx in robot_route_indices if idx < len(robot_route_locations_base)]
                
                if route_points_x: 
                    ax.plot(route_points_x, route_points_y, ".:", color=robot_cmap(j % robot_cmap.N), linewidth=1.5, markersize=4, alpha=0.8, label=f"Robot Route C{cluster_id}-{j+1}" if j==0 else None, zorder=2)

        if last_resort_truck_routes and all_locations_list:
            print(f"[VISUALIZER] Plotting {len(last_resort_truck_routes)} last-resort truck routes.")
            for i, route_indices in enumerate(last_resort_truck_routes):
                if len(route_indices) <= 1: continue
                
                route_actual_locations = [all_locations_list[idx] for idx in route_indices]
                route_x = [loc.x for loc in route_actual_locations]
                route_y = [loc.y for loc in route_actual_locations]

                ax.plot(
                    route_x, route_y, 'x--', color=last_resort_cmap(i % last_resort_cmap.N), 
                    linewidth=1.5, markersize=6, label=f"Last-Resort Truck Route {i+1}", zorder=4
                )
                for loc_idx in route_indices[1:-1]: 
                    ax.scatter(all_locations_list[loc_idx].x, all_locations_list[loc_idx].y, marker='P', color=last_resort_cmap(i % last_resort_cmap.N), s=70, edgecolors='black', zorder=4, label="Last-Resort Customer" if loc_idx == route_indices[1] and i == 0 else None)


        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) 
        ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 1]) 
        return fig
