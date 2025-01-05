class Truck:
    """
    A class representing a delivery truck.
    """

    def __init__(
        self,
        truck_id=0,
        capacity=200,
        speed=60,
        cost_per_km=0.5,
        cost_per_hour=20,
        emissions_per_km=120,
    ):
        """
        Initialize a Truck instance with default or custom properties.

        :param truck_id: Unique ID for the truck.
        :param capacity: Maximum capacity of the truck in units.
        :param speed: Average speed of the truck in km/h.
        :param cost_per_km: Operating cost per kilometer.
        :param cost_per_hour: Operating cost per hour.
        :param emissions_per_km: Carbon emissions per kilometer in grams.
        """
        self.truck_id = truck_id
        self.capacity = capacity
        self.speed = speed
        self.cost_per_km = cost_per_km
        self.cost_per_hour = cost_per_hour
        self.emissions_per_km = emissions_per_km

    def calculate_trip_cost(self, distance, time):
        """
        Calculate the total trip cost based on distance and time.

        :param distance: Total distance traveled in kilometers.
        :param time: Total time taken in hours.
        :return: Total cost in currency units.
        """
        return distance * self.cost_per_km + time * self.cost_per_hour

    def calculate_emissions(self, distance):
        """
        Calculate the total emissions for a trip.

        :param distance: Total distance traveled in kilometers.
        :return: Total emissions in grams.
        """
        return distance * self.emissions_per_km
