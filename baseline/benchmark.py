# vrp_benchmark_runner.py

import sys
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from entities.truck import Truck
from distance_matrix_calculator import DistanceMatrixCalculator
from truck_delivery_baseline import TruckDeliveryBaseline


# Single Responsibility Classes
class SolutionVerifier:
    """Verifies the validity and capacity constraints of a VRP solution."""

    def verify(self, result, demands, distance_matrix, trucks):
        messages = []
        if result["routes"] is None:
            messages.append("No solution to verify.")
            return messages

        messages.append("--- Verification ---")
        visited_customers = set()
        route_demands = []
        route_distances = []

        for idx, route in enumerate(result["routes"]):
            if len(route) <= 2:
                continue  # Skip routes with no customers
            truck_demand = sum(demands[cust] for cust in route[1:-1])
            route_demands.append((idx, truck_demand))

            distance = 0.0
            for i in range(len(route) - 1):
                distance += distance_matrix[route[i]][route[i + 1]]
            route_distances.append((idx, distance))

            for cust in route[1:-1]:
                if cust in visited_customers:
                    messages.append(f"Error: Customer {cust} visited more than once.")
                visited_customers.add(cust)

        all_customers = set(range(1, len(demands)))  # exclude depot (0)
        missing = all_customers - visited_customers
        if missing:
            messages.append(f"Error: Missing customers in routes: {missing}")
        else:
            messages.append("All customers visited exactly once.")

        for idx, demand in route_demands:
            if demand > trucks[idx].capacity:
                messages.append(
                    f"Error: Truck {idx+1} exceeds capacity with demand {demand}."
                )

        return messages


class RouteVisualizer:
    """Visualizes VRP routes on a 2D plot."""

    def plot_routes(self, routes, customers):
        plt.figure(figsize=(10, 10))
        # Plot customers and depot
        plt.scatter(customers["X"], customers["Y"], c="blue", label="Customers")
        depot = customers.iloc[0]
        plt.scatter(depot["X"], depot["Y"], c="red", marker="s", label="Depot")

        # Plot each route
        for route in routes:
            if len(route) <= 2:
                continue  # skip empty routes
            route_coords = customers.iloc[route][["X", "Y"]].values
            plt.plot(route_coords[:, 0], route_coords[:, 1], marker="o")

        plt.title("Truck Delivery Routes")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()


class RouteSaver:
    """Saves VRP routes to a CSV file."""

    def save_routes_to_csv(self, routes, filename="routes.csv"):
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Truck", "Route"])
            for idx, route in enumerate(routes):
                writer.writerow([f"Truck {idx+1}", " -> ".join(map(str, route))])
        print(f"Routes have been saved to '{filename}'.")


class BenchmarkCalculator:
    """Calculates benchmark data such as cost, time, and emissions for VRP routes."""

    def calculate(self, routes, distance_matrix, trucks):
        total_cost = 0.0
        total_emissions = 0.0
        total_time = 0.0

        for idx, route in enumerate(routes):
            distance = 0.0
            for j in range(len(route) - 1):
                distance += distance_matrix[route[j]][route[j + 1]]
            truck = trucks[idx]
            time_needed = distance / truck.speed  # in hours
            cost = distance * truck.cost_per_km + time_needed * truck.cost_per_hour
            emissions = distance * truck.emissions_per_km

            total_cost += cost
            total_emissions += emissions
            total_time += time_needed

        return total_cost, total_time, total_emissions


class ReportSaver:
    """Saves VRP report including routes, benchmark data, and verification summary to a file."""

    def save_report(
        self,
        filename,
        routes,
        total_cost,
        total_time,
        total_emissions,
        verification_messages,
    ):
        with open(filename, "w", encoding="utf-8") as file:
            file.write("--- Routes ---\n")
            for idx, route in enumerate(routes):
                file.write(f"Truck {idx+1}: {' -> '.join(map(str, route))}\n")
            file.write(f"\n--- Benchmark Data ---\n")
            file.write(f"Total Cost: {total_cost:.2f}\n")
            file.write(f"Total Time (sum of all routes in hours): {total_time:.2f}\n")
            file.write(f"Total CO₂ Emissions: {total_emissions:.2f} grams\n")
            file.write("\n--- Verification ---\n")
            for msg in verification_messages:
                file.write(msg + "\n")
        print(f"Report saved to '{filename}'.")


class VRPBenchmarkRunner:
    """Central orchestrator for loading data, solving VRP, verifying, visualizing, and saving results."""

    def __init__(self, file_path, num_trucks):
        self.file_path = file_path
        self.num_trucks = num_trucks

        # Initialize helper components
        self.verifier = SolutionVerifier()
        self.visualizer = RouteVisualizer()
        self.saver = RouteSaver()
        self.benchmark_calculator = BenchmarkCalculator()
        self.report_saver = ReportSaver()

    def run(self):
        # Load and prepare data
        dataset = DistanceMatrixCalculator(self.file_path)
        dataset.load_data()
        distance_matrix = dataset.compute_distance_matrix()

        demands = dataset.get_demands()
        ready_times = dataset.get_ready_times()
        due_times = dataset.get_due_times()
        service_times = dataset.get_service_times()

        trucks = [Truck(truck_id=i) for i in range(self.num_trucks)]

        total_demand = sum(demands)
        total_capacity = sum(truck.capacity for truck in trucks)
        print(f"Total Demand: {total_demand}, Total Capacity: {total_capacity}")
        if total_demand > total_capacity:
            print("Warning: Total demand exceeds total capacity. Exiting.")
            sys.exit(1)

        baseline = TruckDeliveryBaseline(
            distance_matrix, trucks, demands, ready_times, due_times, service_times
        )
        result = baseline.solve_vrp()
        if result["routes"] is None:
            print("No solution found.")
            sys.exit(1)

        # Display routes and total distance
        print("\n--- Routes ---")
        for idx, route in enumerate(result["routes"]):
            print(f"Truck {idx+1}: {route}")
        print(f"\nTotal Distance Traveled: {result['total_distance']}")

        # Calculate benchmark data
        total_cost, total_time, total_emissions = self.benchmark_calculator.calculate(
            result["routes"], distance_matrix, trucks
        )
        print(f"\n--- Benchmark Data ---")
        print(f"Total Cost: {total_cost:.2f}")
        print(f"Total Time (sum of all routes in hours): {total_time:.2f}")
        print(f"Total CO₂ Emissions: {total_emissions:.2f} grams")

        # Verification
        verification_messages = self.verifier.verify(
            result, demands, distance_matrix, trucks
        )
        for msg in verification_messages:
            print(msg)

        # Save routes to CSV
        self.saver.save_routes_to_csv(result["routes"])

        # Save full report
        self.report_saver.save_report(
            filename="vrp_report.txt",
            routes=result["routes"],
            total_cost=total_cost,
            total_time=total_time,
            total_emissions=total_emissions,
            verification_messages=verification_messages,
        )

        # Visualization
        customers_coords = dataset.customers[["X", "Y"]]
        self.visualizer.plot_routes(result["routes"], customers_coords)


# Usage of the Runner class
if __name__ == "__main__":
    file_path = "dataset/c101.txt"
    num_trucks = 25  # Adjust truck count as needed
    runner = VRPBenchmarkRunner(file_path, num_trucks)
    runner.run()
