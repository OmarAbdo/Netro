# netro/infrastructure/io/solomon_reader.py
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from netro.core.entities.location import Location


class SolomonReader:
    """
    Parser for Solomon VRPTW benchmark dataset files.
    """

    def __init__(self, file_path: str):
        """
        Initialize the Solomon dataset reader.

        Args:
            file_path: Path to the Solomon VRPTW instance file.
        """
        self.file_path = file_path
        self.num_vehicles = 0
        self.vehicle_capacity = 0
        self.locations = []

    def read(self) -> Tuple[List[Location], int, float]:
        """
        Read and parse the Solomon dataset.

        Returns:
            A tuple containing:
            - List of Location objects (depot is at index 0).
            - Number of vehicles specified in the dataset.
            - Vehicle capacity specified in the dataset.
        """
        # First, parse the file to extract header info and raw data
        with open(self.file_path, "r") as file:
            lines = file.readlines()

        # Find vehicle info
        vehicle_info_found = False
        customer_data_start_idx = 0

        for idx, line in enumerate(lines):
            if line.strip().startswith("NUMBER"):
                # The next line contains the numbers
                number_line = lines[idx + 1]
                parts = number_line.strip().split()
                if len(parts) >= 2:
                    self.num_vehicles = int(parts[0])
                    self.vehicle_capacity = float(parts[1])
                    vehicle_info_found = True
                    customer_data_start_idx = idx + 2
                    break

        if not vehicle_info_found:
            raise ValueError("Vehicle information not found in the dataset")

        # Find the header line for customer data
        for idx in range(customer_data_start_idx, len(lines)):
            if lines[idx].strip().startswith("CUST NO"):
                customer_data_start_idx = idx + 1
                break

        # Parse customer data
        locations = []

        for line in lines[customer_data_start_idx:]:
            parts = line.strip().split()
            if len(parts) >= 7 and parts[0].isdigit():
                try:
                    locations.append(
                        Location(
                            id=int(parts[0]),
                            x=float(parts[1]),
                            y=float(parts[2]),
                            demand=float(parts[3]),
                            ready_time=float(parts[4]),
                            due_time=float(parts[5]),
                            service_time=float(parts[6]),
                        )
                    )
                except ValueError:
                    continue  # Skip lines with parsing errors

        self.locations = locations
        return locations, self.num_vehicles, self.vehicle_capacity
