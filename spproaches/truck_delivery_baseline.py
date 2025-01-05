from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np


class TruckDeliveryBaseline:
    """
    Baseline implementation for traditional truck-based delivery using OR-Tools.
    """

    def __init__(self, distance_matrix, vehicle_capacity, demands, num_vehicles):
        """
        Initialize the baseline class.

        :param distance_matrix: 2D numpy array representing the distance matrix.
        :param vehicle_capacity: Capacity of each vehicle.
        :param demands: List of customer demands.
        :param num_vehicles: Number of vehicles available for delivery.
        """
        self.distance_matrix = distance_matrix
        self.vehicle_capacity = vehicle_capacity
        self.demands = demands
        self.num_vehicles = num_vehicles

    def create_data_model(self):
        """Create data model for OR-Tools VRP solver."""
        data = {
            "distance_matrix": self.distance_matrix,
            "demands": self.demands,
            "vehicle_capacities": [self.vehicle_capacity] * self.num_vehicles,
            "num_vehicles": self.num_vehicles,
            "depot": 0,  # Depot is the starting location
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
        :return: Dictionary containing the routes and total distance.
        """
        routes = []
        total_distance = 0
        for vehicle_id in range(self.num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            routes.append(route)
        return {"routes": routes, "total_distance": total_distance}


# Example Usage
if __name__ == "__main__":
    # Example inputs
    demands = [
        0,
        10,
        30,
        10,
        20,
        20,
        10,
        20,
        10,
        10,
    ]  # Example demands for each customer
    vehicle_capacity = 200  # Vehicle capacity
    num_vehicles = 2  # Number of vehicles

    # Load distance matrix (from previous steps)
    distance_matrix = np.random.rand(10, 10)  # Replace with actual distance matrix

    # Initialize and solve the VRP
    baseline = TruckDeliveryBaseline(
        distance_matrix, vehicle_capacity, demands, num_vehicles
    )
    result = baseline.solve_vrp()

    # Display the results
    print("Routes:", result["routes"])
    print("Total Distance:", result["total_distance"])
