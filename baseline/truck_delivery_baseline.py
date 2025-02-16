# truck_delivery_baseline.py

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tqdm import tqdm
import time
import sys
import os

sys.path.append(os.getcwd())

from entities.truck import Truck  # Import Truck class
from distance_matrix_calculator import (
    DistanceMatrixCalculator,
)  # Import DistanceMatrixCalculator


class TruckDeliveryBaseline:
    """
    Baseline implementation for traditional truck-based delivery using OR-Tools.
    """

    def __init__(
        self, distance_matrix, trucks, demands, ready_times, due_times, service_times
    ):
        """
        Initialize the baseline class.

        :param distance_matrix: 2D numpy array representing the distance matrix.
        :param trucks: List of Truck objects.
        :param demands: List of customer demands.
        :param ready_times: List of customer ready times (unused in CVRP).
        :param due_times: List of customer due times (unused in CVRP).
        :param service_times: List of customer service times (unused in CVRP).
        """
        self.distance_matrix = distance_matrix
        self.trucks = trucks
        self.demands = demands
        self.ready_times = ready_times  # Unused
        self.due_times = due_times  # Unused
        self.service_times = service_times  # Unused
        self.num_vehicles = len(trucks)

    def create_data_model(self):
        """Create data model for OR-Tools VRP solver."""
        data = {
            "distance_matrix": self.distance_matrix,
            "demands": self.demands,
            "vehicle_capacities": [truck.capacity for truck in self.trucks],
            "num_vehicles": self.num_vehicles,
            "depot": 0,  # Depot is the starting location
            # Time-related data is excluded
        }
        return data

    def solve_vrp(self):
        """
        Solve the Capacitated Vehicle Routing Problem using OR-Tools.

        :return: A dictionary containing the routes and total distance.
        """
        print("Starting the solver...")
        start_time = time.time()
        data = self.create_data_model()
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # Define the cost function (distance)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(data["distance_matrix"][from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # Null slack
            data["vehicle_capacities"],  # Vehicle maximum capacities
            True,  # Start cumul to zero
            "Capacity",
        )

        # Define search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 300  # 5-minute limit
        search_parameters.log_search = True  # Enable progress logs

        # Choose a good first-solution strategy
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Choose a local-search metaheuristic
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

        # Enable additional logging options if available
        # Note: 'trace_propagation' is invalid and thus removed
        # You can use 'search_parameters.solution_limit' to limit the number of solutions considered
        search_parameters.solution_limit = 1000  # Limit on solutions to consider

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        end_time = time.time()

        if solution:
            print(f"Solver completed in {end_time - start_time:.2f} seconds.")
            return self.extract_solution(manager, routing, solution)
        else:
            print("Solver failed to find a solution within the time limit.")
            return {"routes": None, "total_distance": None}

    def extract_solution(self, manager, routing, solution):
        """
        Extract the solution from the OR-Tools solver.

        :param manager: RoutingIndexManager object.
        :param routing: RoutingModel object.
        :param solution: Solution object.
        :return: Dictionary containing the routes and total distance.
        """
        print("Extracting solution...")
        routes = []
        total_distance = 0

        for vehicle_id in tqdm(range(self.num_vehicles), desc="Processing vehicles"):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            # Add the depot at the end
            route.append(manager.IndexToNode(index))
            total_distance += route_distance
            routes.append(route)

        print("Solution extraction complete.")
        return {"routes": routes, "total_distance": total_distance}


# Example Usage
if __name__ == "__main__":
    # Path to the Solomon VRPTW instance file
    file_path = "dataset/c101.txt"

    # Initialize the DistanceMatrixCalculator
    dataset = DistanceMatrixCalculator(file_path)

    # Load the dataset
    print("Step 1: Loading the dataset...")
    dataset.load_data()
    print("Dataset loaded successfully!")

    # Compute the distance matrix
    print("Step 2: Computing the distance matrix...")
    distance_matrix = dataset.compute_distance_matrix()
    print("Distance matrix computed!")

    # Extract specific data
    print("Step 3: Extracting additional data...")
    demands = dataset.get_demands()
    # Time-related data is not used in CVRP
    ready_times = dataset.get_ready_times()
    due_times = dataset.get_due_times()
    service_times = dataset.get_service_times()

    # Define trucks
    num_trucks = 25  # Updated number of trucks to match the dataset
    trucks = [Truck(truck_id=i) for i in range(num_trucks)]

    # 4.1 Insert Verification Code Here
    # ----------------------------------
    # Verify feasibility
    total_demand = sum(demands)
    total_capacity = sum(truck.capacity for truck in trucks)
    print(f"Total Demand: {total_demand}")
    print(f"Total Capacity: {total_capacity}")

    if total_demand > total_capacity:
        print(
            "Warning: Total demand exceeds total capacity. The problem may be infeasible."
        )
    else:
        print("Total demand is within total capacity.")
    # ----------------------------------

    # Initialize and solve the VRP
    baseline = TruckDeliveryBaseline(
        distance_matrix, trucks, demands, ready_times, due_times, service_times
    )
    result = baseline.solve_vrp()

    # Display detailed results
    if result["routes"] is not None:
        print("\n--- Routes ---")
        for idx, route in enumerate(result["routes"]):
            print(f"Truck {idx + 1}: {route}")
        print(f"\nTotal Distance Traveled: {result['total_distance']}")
    else:
        print("No solution found.")

    # Display detailed results
    if result["routes"] is not None:
        print("\n--- Routes ---")
        for idx, route in enumerate(result["routes"]):
            print(f"Truck {idx + 1}: {route}")
        print(f"\nTotal Distance Traveled: {result['total_distance']}")

        # Additional Benchmarking
        total_cost = 0.0
        total_emissions = 0.0
        total_time = 0.0

        for idx, route in enumerate(result["routes"]):
            distance = 0.0
            # Calculate distance for the current route using the distance matrix
            for j in range(len(route) - 1):
                from_node = route[j]
                to_node = route[j + 1]
                distance += distance_matrix[from_node][to_node]

            # Use the corresponding truck for route idx
            truck = trucks[idx]

            # Compute time required, cost, and emissions for this route
            time_needed = distance / truck.speed  # in hours
            cost = distance * truck.cost_per_km + time_needed * truck.cost_per_hour
            emissions = distance * truck.emissions_per_km

            total_cost += cost
            total_emissions += emissions
            total_time += time_needed

        print(f"\n--- Benchmark Data ---")
        print(f"Total Cost: {total_cost:.2f}")
        print(f"Total Time (sum of all routes in hours): {total_time:.2f}")
        print(f"Total CO₂ Emissions: {total_emissions:.2f} grams")
    else:
        print("No solution found.")
