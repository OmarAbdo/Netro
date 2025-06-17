# netro/application.py
from typing import Dict, List, Any, Optional, Tuple
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Core entities - updated imports with 'netro.' prefix
from netro.core.entities.location import Location
from netro.core.entities.vehicle import Truck, Robot
from netro.core.entities.cluster import Cluster

# Services - updated imports with 'netro.' prefix
from netro.services.clustering.hdbscan_clustering import HDBSCANClusterer
from netro.services.clustering.capacitated_splitter import CapacitatedClusterSplitter
from netro.services.clustering.cluster_service import ClusterService
from netro.services.routing.ortools_cvrp import ORToolsCVRP
from netro.services.routing.local_search import AdaptiveLocalSearch
from netro.services.routing.robot_routing_service import RobotRoutingService
from netro.services.routing.netro_routing_service import NetroRoutingService
from netro.services.benchmarking.baseline_service import BaselineTruckService
from netro.services.benchmarking.comparison_service import ComparisonService
from netro.services.robot_deployment.smart_loader import SmartLoader

# Infrastructure - updated imports with 'netro.' prefix
from netro.infrastructure.io.solomon_reader import SolomonReader
from netro.infrastructure.visualization.solution_visualizer import SolutionVisualizer

# Configuration - updated imports with 'netro.' prefix
from netro.config.env import get_config


# Rest of the code remains the same
class NetroApplication:
    """
    Main application class for the Netro last-mile delivery optimization system.

    This class orchestrates the complete process:
    1. Load and prepare data
    2. Initialize services
    3. Run baseline (truck-only) solution
    4. Run Netro (truck + robot) solution
    5. Compare and visualize results
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the application.

        Args:
            config_path: Optional path to a config file. If None, default config is used.
        """
        # Load configuration
        self.config = get_config()

        # Create output directory if it doesn't exist
        output_path = Path(self.config["IO"]["output_path"])
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize attributes
        self.dataset_path = None
        self.locations = []
        self.depot = None
        self.trucks = []
        self.robots = []
        self.clusters = []
        self.centroids = {}

        # Solution results
        self.baseline_solution = None
        self.netro_solution = None
        self.comparison = None

    def load_dataset(self, dataset_name: str) -> None:
        """
        Load a dataset from the configured dataset directory.

        Args:
            dataset_name: Name of the dataset file (e.g., 'c101.txt').
        """
        dataset_path = os.path.join(self.config["IO"]["dataset_path"], dataset_name)
        self.dataset_path = dataset_path

        print(f"Loading dataset: {dataset_path}")
        reader = SolomonReader(dataset_path)
        self.locations, num_vehicles, vehicle_capacity = reader.read()

        # Override truck capacity if specified in dataset
        self.config["VEHICLES"]["truck"]["capacity"] = vehicle_capacity

        # Set depot
        if self.locations:
            self.depot = self.locations[0]
            print(
                f"Depot: ID={self.depot.id}, Location=({self.depot.x:.1f}, {self.depot.y:.1f})"
            )
            print(f"Number of customers: {len(self.locations) - 1}")
            print(f"Total demand: {sum(loc.demand for loc in self.locations[1:]):.1f}")
            print(f"Vehicle capacity: {vehicle_capacity}")
        else:
            print("Error: No locations loaded from dataset")

    def initialize_vehicles(self) -> None:
        """
        Initialize trucks and robots based on configuration.
        """
        # Create trucks
        truck_config = self.config["VEHICLES"]["truck"]
        # Calculate number of trucks needed based on total demand
        num_trucks = max(
            1,
            int(
                np.ceil(
                    sum(loc.demand for loc in self.locations[1:])
                    / truck_config["capacity"]
                )
            ),
        )

        self.trucks = [
            Truck(
                id=i,
                capacity=truck_config["capacity"],
                speed=truck_config["speed"],
                cost_per_distance=truck_config["cost_per_distance"],
                cost_per_time=truck_config["cost_per_time"],
                emissions_per_distance=truck_config["emissions_per_distance"],
                robot_capacity=truck_config["robot_capacity"],
                loading_time=truck_config["loading_time"],
            )
            for i in range(num_trucks)
        ]

        # Create robots
        robot_config = self.config["VEHICLES"]["robot"]
        robots_per_truck = truck_config["robot_capacity"]
        total_robots = num_trucks * robots_per_truck

        self.robots = [
            Robot(
                id=i,
                capacity=robot_config["capacity"],
                speed=robot_config["speed"],
                cost_per_distance=robot_config["cost_per_distance"],
                cost_per_time=robot_config["cost_per_time"],
                emissions_per_distance=robot_config["emissions_per_distance"],
                battery_capacity=robot_config["battery_capacity"],
                recharging_rate=robot_config["recharging_rate"],
            )
            for i in range(total_robots)
        ]

        print(f"Created {len(self.trucks)} trucks and {len(self.robots)} robots")

    def run_clustering(self) -> None:
        """
        Run customer clustering and capacitated splitting.
        """
        if not self.locations or not self.depot:
            raise ValueError("Dataset must be loaded before clustering")

        print("Running customer clustering...")

        # Create clustering services
        clustering_config = self.config["CLUSTERING"]
        clusterer = HDBSCANClusterer(
            min_cluster_size=clustering_config["min_cluster_size"]
        )
        splitter = CapacitatedClusterSplitter()

        cluster_service = ClusterService(clusterer, splitter)

        # Customer locations (excluding depot)
        customer_locations = self.locations[1:]

        # Run clustering
        truck_config = self.config["VEHICLES"]["truck"]
        robot_config = self.config["VEHICLES"]["robot"]
        
        num_robots_per_truck = truck_config.get("robot_capacity", 1) 
        capacity_per_robot = robot_config.get("capacity", 30) 
        
        effective_cluster_capacity = num_robots_per_truck * capacity_per_robot
        print(f"[INFO] Effective capacity for cluster splitting (1 truck's robots): {num_robots_per_truck} robots * {capacity_per_robot} cap/robot = {effective_cluster_capacity}")

        self.clusters, self.centroids = cluster_service.cluster_locations(
            locations=customer_locations,
            truck_capacity=effective_cluster_capacity, 
        )

        print(f"Created {len(self.clusters)} clusters")
        for cluster in self.clusters:
            print(
                f"  Cluster {cluster.id}: {len(cluster.locations)} customers, "
                f"demand={sum(loc.demand for loc in cluster.locations):.1f}"
            )

    def run_baseline(self) -> Dict[str, Any]:
        """
        Run the baseline truck-only delivery solution.

        Returns:
            Solution dictionary.
        """
        if not self.locations or not self.depot:
            raise ValueError("Dataset must be loaded before running baseline")

        print("Running baseline truck-only solution...")

        cvrp_config = self.config["ROUTING"]["cvrp"]
        main_cvrp_solver = ORToolsCVRP(
            first_solution_strategy=cvrp_config["first_solution_strategy"],
            local_search_metaheuristic=cvrp_config["local_search_metaheuristic"],
            time_limit_seconds=cvrp_config["time_limit_seconds"],
        )

        baseline_service = BaselineTruckService(main_cvrp_solver)

        start_time = time.time()
        self.baseline_solution = baseline_service.solve(
            depot=self.depot, customers=self.locations[1:], trucks=self.trucks
        )
        end_time = time.time()

        print(f"Baseline solution computed in {end_time - start_time:.2f} seconds")
        print(f"Total distance: {self.baseline_solution['total_distance']:.2f} km")
        print(f"Total time: {self.baseline_solution['total_time']:.2f} hours")
        print(f"Total cost: {self.baseline_solution['total_cost']:.2f}")
        print(f"Number of routes: {len(self.baseline_solution['routes'])}")

        return self.baseline_solution

    def run_netro(self) -> Dict[str, Any]:
        """
        Run the Netro truck-robot hybrid solution, ensuring all customers are served,
        using last-resort truck routes if necessary.

        Returns:
            Solution dictionary.
        """
        if not self.clusters or not self.centroids:
            raise ValueError("Clustering must be run before Netro")

        print("Running Netro truck-robot hybrid solution...")

        cvrp_config = self.config["ROUTING"]["cvrp"]
        
        truck_router = ORToolsCVRP(
            first_solution_strategy=cvrp_config["first_solution_strategy"],
            local_search_metaheuristic=cvrp_config["local_search_metaheuristic"],
            time_limit_seconds=cvrp_config["time_limit_seconds"],
        )

        robot_cvrp_config = self.config["ROUTING"].get("robot_cvrp", cvrp_config)
        robot_router_time_limit = robot_cvrp_config.get("time_limit_seconds_robot", 60)
        
        print(f"[INFO] Robot OR-Tools solver time limit: {robot_router_time_limit}s")

        robot_solver_instance = ORToolsCVRP(
            first_solution_strategy=robot_cvrp_config.get("first_solution_strategy", "PATH_CHEAPEST_ARC"),
            local_search_metaheuristic=robot_cvrp_config.get("local_search_metaheuristic_robot", "AUTOMATIC"),
            time_limit_seconds=robot_router_time_limit,
        )
        print(f"[INFO] Robot OR-Tools local search metaheuristic: {robot_solver_instance.local_search_metaheuristic}")

        deployment_config = self.config["DEPLOYMENT"]
        robot_routing_service = RobotRoutingService(
            routing_algorithm=robot_solver_instance,
            recharge_time_factor=deployment_config["recharge_time_factor"],
            robot_launch_time=deployment_config["robot_launch_time"],
            robot_recovery_time=deployment_config["robot_recovery_time"],
        )

        netro_service = NetroRoutingService(
            truck_routing_algorithm=truck_router,
            robot_routing_service=robot_routing_service,
            robot_unloading_time=deployment_config["loading_time_per_robot"]
            * self.config["VEHICLES"]["truck"]["robot_capacity"],
        )

        robots_per_truck = []
        robot_fleet_capacity_per_truck = self.config["VEHICLES"]["truck"]["robot_capacity"]
        robot_idx = 0
        for _ in range(len(self.trucks)):
            truck_robots = []
            for _ in range(robot_fleet_capacity_per_truck):
                if robot_idx < len(self.robots):
                    truck_robots.append(self.robots[robot_idx])
                    robot_idx += 1
            robots_per_truck.append(truck_robots)

        start_time = time.time()
        self.netro_solution = netro_service.solve(
            depot=self.depot,
            clusters=self.clusters,
            centroids=self.centroids,
            trucks=self.trucks,
            robots_per_truck=robots_per_truck,
        )
        end_time = time.time()
        
        print(f"Initial Netro solution computed in {end_time - start_time:.2f} seconds")

        # --- BEGIN LAST-RESORT TRUCK ROUTING FOR UNSERVED CUSTOMERS ---
        all_dataset_customers = self.locations[1:] 
        
        served_customer_ids = set() # Initialize the set here
        if 'cluster_routes' in self.netro_solution and self.netro_solution['cluster_routes']:
            cluster_id_to_locations_map = {c.id: c.locations for c in self.clusters}
            
            for cluster_identifier, robot_routes_in_cluster in self.netro_solution['cluster_routes'].items():
                # Assuming cluster_identifier is cluster.id (integer)
                # If it's a string like "x_y", this lookup needs to be adapted
                cluster_customer_list = cluster_id_to_locations_map.get(cluster_identifier)
                
                if cluster_customer_list:
                    for robot_route in robot_routes_in_cluster:
                        for customer_node_idx in robot_route[1:-1]: 
                            original_customer_obj = cluster_customer_list[customer_node_idx - 1]
                            served_customer_ids.add(original_customer_obj.id)
                else:
                    print(f"[WARNING] Could not find customer list for cluster_identifier: {cluster_identifier} in last-resort routing.")

        unserved_customers = [cust for cust in all_dataset_customers if cust.id not in served_customer_ids]

        if unserved_customers:
            print(f"[INFO] {len(unserved_customers)} customers initially unserved by Netro hybrid. Routing with last-resort trucks...")
            
            last_resort_cvrp_solver = ORToolsCVRP(
                first_solution_strategy=cvrp_config["first_solution_strategy"],
                local_search_metaheuristic=cvrp_config["local_search_metaheuristic"],
                time_limit_seconds=cvrp_config.get("time_limit_seconds_last_resort", 30),
            )
            last_resort_truck_service = BaselineTruckService(last_resort_cvrp_solver)
            
            last_resort_solution = last_resort_truck_service.solve(
                depot=self.depot, 
                customers=unserved_customers, 
                trucks=self.trucks 
            )

            if last_resort_solution and last_resort_solution.get('routes'):
                num_newly_served = len(unserved_customers) - len(last_resort_solution.get('unserved_customers', []))
                print(f"[INFO] Last-resort trucks served {num_newly_served} additional customers.")
                
                self.netro_solution['last_resort_truck_routes'] = last_resort_solution['routes']
                self.netro_solution['total_truck_distance'] += last_resort_solution['total_distance']
                
                last_resort_time = last_resort_solution['total_time']
                self.netro_solution['last_resort_truck_time'] = last_resort_time
                
                # How to combine time is complex. For now, let's assume last_resort runs sequentially after the longest Netro leg.
                # This might not be optimal for "total parallel time" but ensures all work is accounted for.
                # A more sophisticated model might try to run these in parallel if trucks are available.
                self.netro_solution['total_time_with_last_resort'] = self.netro_solution['total_time'] + last_resort_time
                
                if 'total_cost' in last_resort_solution:
                    self.netro_solution['total_cost'] = self.netro_solution.get('total_cost', 0) + last_resort_solution['total_cost']
                if 'total_emissions' in last_resort_solution:
                     self.netro_solution['total_emissions'] = self.netro_solution.get('total_emissions', 0) + last_resort_solution['total_emissions']

                # Add the last resort routes to the main truck routes for visualization if needed,
                # but be mindful this might affect other calculations if not handled carefully.
                # For now, keeping them separate in the solution dict is safer.
                # self.netro_solution['truck_routes'].extend(last_resort_solution['routes'])
            else:
                print("[INFO] No last-resort truck routes generated or all unserved customers remained unserved by last-resort.")
                self.netro_solution['last_resort_truck_routes'] = []
                self.netro_solution['last_resort_truck_time'] = 0.0
                self.netro_solution['total_time_with_last_resort'] = self.netro_solution['total_time']
        else:
            print("[INFO] All customers served by initial Netro hybrid solution. No last-resort trucks needed.")
            self.netro_solution['last_resort_truck_routes'] = []
            self.netro_solution['last_resort_truck_time'] = 0.0
            self.netro_solution['total_time_with_last_resort'] = self.netro_solution['total_time']
        # --- END LAST-RESORT TRUCK ROUTING ---

        print(f"Final Netro solution (including last-resort if any):")
        print(f"  Total truck distance: {self.netro_solution['total_truck_distance']:.2f} km")
        print(f"  Total robot distance: {self.netro_solution['total_robot_distance']:.2f} km")
        print(f"  Netro parallel time: {self.netro_solution['total_time']:.2f} hours")
        if 'last_resort_truck_time' in self.netro_solution and self.netro_solution['last_resort_truck_time'] > 0:
            print(f"  Last-resort truck time (sequential): {self.netro_solution['last_resort_truck_time']:.2f} hours")
            print(f"  Combined total time (Netro + sequential last-resort): {self.netro_solution['total_time_with_last_resort']:.2f} hours")
        if 'total_cost' in self.netro_solution:
            print(f"  Total cost: {self.netro_solution['total_cost']:.2f}")
        
        return self.netro_solution

    def run_comparison(self) -> Dict[str, Any]:
        """
        Compare baseline and Netro solutions.

        Returns:
            Comparison dictionary.
        """
        if not self.baseline_solution or not self.netro_solution:
            raise ValueError(
                "Both baseline and Netro solutions must be run before comparison"
            )

        print("Comparing solutions...")

        comparison_service = ComparisonService()
        self.comparison = comparison_service.compare(
            baseline_solution=self.baseline_solution, netro_solution=self.netro_solution
        )

        print(comparison_service.generate_report(self.comparison))

        return self.comparison

    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """
        Visualize clustering and routing solutions.

        Args:
            save_path: Optional path to save visualizations. If None, figures are displayed.
        """
        visualizer = SolutionVisualizer()

        if self.clusters:
            print("Visualizing clusters...")
            fig = visualizer.plot_clusters(
                clusters=self.clusters,
                depot=self.depot,
                centroids=self.centroids,
                title="Customer Clusters",
            )
            if save_path:
                fig.savefig(
                    os.path.join(save_path, "clusters.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

        if self.baseline_solution and "routes" in self.baseline_solution:
            print("Visualizing baseline solution...")
            fig = visualizer.plot_truck_routes(
                routes=self.baseline_solution["routes"],
                locations=self.locations,
                title="Baseline Truck-Only Routes",
            )
            if save_path:
                fig.savefig(
                    os.path.join(save_path, "baseline_routes.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

        if self.netro_solution and "truck_routes" in self.netro_solution:
            print("Visualizing Netro solution...")
            # Include last_resort_truck_routes if they exist
            last_resort_routes = self.netro_solution.get('last_resort_truck_routes', [])
            
            fig = visualizer.plot_netro_solution(
                depot=self.depot,
                clusters=self.clusters,
                truck_routes=self.netro_solution["truck_routes"],
                cluster_routes=self.netro_solution["cluster_routes"],
                centroids=self.centroids,
                title="Netro Hybrid Truck-Robot Solution",
                last_resort_truck_routes=last_resort_routes, # Pass to visualizer
                all_locations_list=self.locations # Pass the full locations list
            )
            if save_path:
                fig.savefig(
                    os.path.join(save_path, "netro_solution.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

        if self.comparison:
            print("Visualizing comparison...")
            comparison_service = ComparisonService()
            fig = comparison_service.plot_comparison(self.comparison)
            if save_path:
                fig.savefig(
                    os.path.join(save_path, "comparison.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()

    def run_full_workflow(
        self, dataset_name: str, save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete workflow: dataset loading, clustering, baseline, Netro, and comparison.

        Args:
            dataset_name: Name of the dataset file (e.g., 'c101.txt').
            save_visualizations: Whether to save visualizations to output directory.

        Returns:
            Comparison dictionary.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dataset_base = os.path.splitext(dataset_name)[0]
        output_dir = os.path.join(
            self.config["IO"]["output_path"], f"{dataset_base}_{timestamp}"
        )

        if save_visualizations:
            os.makedirs(output_dir, exist_ok=True)

        self.load_dataset(dataset_name)
        self.initialize_vehicles()
        self.run_clustering()
        self.run_baseline()
        self.run_netro()
        comparison = self.run_comparison()

        if save_visualizations:
            self.visualize_results(save_path=output_dir)
            comparison_service = ComparisonService() # Already instantiated in run_comparison
            report = comparison_service.generate_report(self.comparison) # Use self.comparison
            report_path = os.path.join(output_dir, "report.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"Report saved to {report_path}")


        return comparison
