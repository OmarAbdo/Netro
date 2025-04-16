"""
SmartLoader Module
--------------------
This module implements the SmartLoader class that is responsible for the truck smart loading system.
It automatically sorts orders into compartments for efficient deployment of Autonomous Delivery Robots (ADRs).

Inspiration and References:
    - Simoni et al., "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E, p. 67 (lines 15-20).
    - Mourelo Ferrandez et al., "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm", JIEM, p. 377 (lines 10-15).
"""


class SmartLoader:
    """
    SmartLoader class assigns orders to truck compartments using a First-Fit Decreasing (FFD) strategy.

    The algorithm:
    1. Sorts orders in descending order of demand.
       (Inspired by bin-packing heuristics; see Simoni et al. (2020), p. 68, lines 5-10)
    2. Iteratively places each order into the first compartment that has sufficient capacity.
       (See Mourelo Ferrandez et al. (2016), p. 377, lines 20-25)

    This design follows the Single Responsibility Principle, with the class focusing solely on loading logic.
    """

    def __init__(self, compartments):
        """
        Initialize the SmartLoader with the truck's compartments.

        :param compartments: List of compartment capacities.
        """
        self.compartments = compartments

    def load_orders(self, orders):
        """
        Load orders into compartments using a greedy First-Fit Decreasing algorithm.

        Each order is a dictionary with keys: 'id' and 'demand'.

        :param orders: List of order dictionaries.
        :return: A dictionary with:
                 - "assignment": Mapping of compartment index to list of assigned order IDs.
                 - "unassigned": List of order IDs that could not be assigned.
                 - "load": Final load in each compartment.
        """
        # Sort orders by decreasing demand (FFD strategy)
        sorted_orders = sorted(orders, key=lambda x: x["demand"], reverse=True)

        # Initialize tracking for compartment loads and assignments
        compartments_load = {i: 0 for i in range(len(self.compartments))}
        compartments_assignment = {i: [] for i in range(len(self.compartments))}
        unassigned_orders = []

        # Greedily assign each order to the first compartment with enough remaining capacity
        for order in sorted_orders:
            assigned = False
            for i in range(len(self.compartments)):
                if compartments_load[i] + order["demand"] <= self.compartments[i]:
                    compartments_assignment[i].append(order["id"])
                    compartments_load[i] += order["demand"]
                    assigned = True
                    break
            if not assigned:
                unassigned_orders.append(order["id"])

        return {
            "assignment": compartments_assignment,
            "unassigned": unassigned_orders,
            "load": compartments_load,
        }


if __name__ == "__main__":
    # Usage Example for SmartLoader
    # Define compartments with capacities (e.g., three compartments)
    compartments = [100, 150, 120]

    # Define sample orders as dictionaries with 'id' and 'demand'
    orders = [
        {"id": 1, "demand": 80},
        {"id": 2, "demand": 50},
        {"id": 3, "demand": 70},
        {"id": 4, "demand": 90},
        {"id": 5, "demand": 60},
        {"id": 6, "demand": 30},
        {"id": 7, "demand": 40},
        {"id": 8, "demand": 110},
    ]

    # Create a SmartLoader instance and load orders
    loader = SmartLoader(compartments)
    result = loader.load_orders(orders)

    print("Assignment of Orders to Compartments:")
    print(result["assignment"])
    print("Unassigned Orders:")
    print(result["unassigned"])
    print("Final Load in Compartments:")
    print(result["load"])
