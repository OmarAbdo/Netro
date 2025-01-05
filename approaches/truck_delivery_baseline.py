import numpy as np
import sys
import os
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

sys.path.append(os.getcwd())
from entities.truck import Truck  # Import the Truck class


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
        :param ready_times: List of customer ready times.
        :param due_times: List of customer due times.
        :param service_times: List of customer service times.
        """
        self.distance_matrix = distance_matrix
        self.trucks = trucks
        self.demands = demands
        self.ready_times = ready_times
        self.due_times = due_times
        self.service_times = service_times
        self.num_vehicles = len(trucks)

    def create_data_model(self):
        """Create data model for OR-Tools VRP solver."""
        data = {
            "distance_matrix": self.distance_matrix,
            "demands": self.demands,
            "vehicle_capacities": [truck.capacity for truck in self.trucks],
            "num_vehicles": self.num_vehicles,
            "depot": 0,  # Depot is the starting location
            "ready_times": self.ready_times,
            "due_times": self.due_times,
            "service_times": self.service_times,
        }
        return data

    def solve_vrp(self):
        """
        Solve the Vehicle Routing Problem using OR-Tools.

        :return: A dictionary containing the routes and total distance.
        """
        data = self.create_data_model()
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        # Define the cost function (distance)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraints
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

        # Add time window constraints
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][
                to_node
            ]  # Distance as proxy for time

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        time = routing.AddDimension(
            time_callback_index,
            30,  # Slack time
            1440,  # Maximum time (e.g., 24 hours in minutes)
            False,  # Don't force start cumul to zero
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # Add time windows for each location
        for location_idx in range(len(data["ready_times"])):
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(
                data["ready_times"][location_idx], data["due_times"][location_idx]
            )

        # Define search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        # Extract solution
        if solution:
            return self.extract_solution(manager, routing, solution)
        else:
            return {"routes": None, "total_distance": None}

    def extract_solution(self, manager, routing, solution):
        """
        Extract the solution from the OR-Tools solver.

        :param manager: RoutingIndexManager object.
        :param routing: RoutingModel object.
        :param solution: Solution object.
        :return: Dictionary containing the routes, total distance, cost, and emissions.
        """
        routes = []
        total_distance = 0
        total_cost = 0
        total_emissions = 0

        for vehicle_id, truck in enumerate(self.trucks):
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

            total_distance += route_distance
            total_cost += truck.calculate_trip_cost(
                route_distance, route_distance / truck.speed
            )
            total_emissions += truck.calculate_emissions(route_distance)
            routes.append(route)

        return {
            "routes": routes,
            "total_distance": total_distance,
            "total_cost": total_cost,
            "total_emissions": total_emissions,
        }


# Example Usage
if __name__ == "__main__":
    # Example inputs
    distance_matrix = np.random.rand(10, 10)  # Replace with actual distance matrix
    demands = [0, 10, 30, 10, 20, 20, 10, 20, 10, 10]
    ready_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    due_times = [1236, 50, 60, 146, 67, 702, 225, 324, 410, 505]
    service_times = [0, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    # Define trucks
    trucks = [Truck(truck_id=i) for i in range(2)]

    # Initialize and solve the VRP
    baseline = TruckDeliveryBaseline(
        distance_matrix, trucks, demands, ready_times, due_times, service_times
    )
    result = baseline.solve_vrp()

    # Display detailed results
    print("Routes:", result["routes"])
    print("Total Distance Traveled:", result["total_distance"])
    print("Total Cost:", result["total_cost"])
    print("Total Emissions (grams):", result["total_emissions"])
