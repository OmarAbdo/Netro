# netro/config/env.py
"""
Environment-specific configuration that overrides default settings.
Values can be loaded from environment variables or a .env file.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from .default import CLUSTERING, VEHICLES, ROUTING, DEPLOYMENT, IO

# Load environment variables from .env file if it exists
load_dotenv()


def get_config() -> Dict[str, Any]:
    """
    Get configuration with environment overrides.

    Returns:
        Dictionary with merged configuration.
    """
    # Start with default config
    config = {
        "CLUSTERING": CLUSTERING,
        "VEHICLES": VEHICLES,
        "ROUTING": ROUTING,
        "DEPLOYMENT": DEPLOYMENT,
        "IO": IO,
    }

    # Override with environment variables
    # CLUSTERING
    if "NETRO_MIN_CLUSTER_SIZE" in os.environ:
        config["CLUSTERING"]["min_cluster_size"] = int(
            os.environ["NETRO_MIN_CLUSTER_SIZE"]
        )

    if "NETRO_CLUSTERING_ALGORITHM" in os.environ:
        config["CLUSTERING"]["algorithm"] = os.environ["NETRO_CLUSTERING_ALGORITHM"]

    # VEHICLES - TRUCK
    if "NETRO_TRUCK_CAPACITY" in os.environ:
        config["VEHICLES"]["truck"]["capacity"] = float(
            os.environ["NETRO_TRUCK_CAPACITY"]
        )

    if "NETRO_TRUCK_SPEED" in os.environ:
        config["VEHICLES"]["truck"]["speed"] = float(os.environ["NETRO_TRUCK_SPEED"])

    if "NETRO_TRUCK_ROBOT_CAPACITY" in os.environ:
        config["VEHICLES"]["truck"]["robot_capacity"] = int(
            os.environ["NETRO_TRUCK_ROBOT_CAPACITY"]
        )

    # VEHICLES - ROBOT
    if "NETRO_ROBOT_CAPACITY" in os.environ:
        config["VEHICLES"]["robot"]["capacity"] = float(
            os.environ["NETRO_ROBOT_CAPACITY"]
        )

    if "NETRO_ROBOT_SPEED" in os.environ:
        config["VEHICLES"]["robot"]["speed"] = float(os.environ["NETRO_ROBOT_SPEED"])

    if "NETRO_ROBOT_BATTERY_CAPACITY" in os.environ:
        config["VEHICLES"]["robot"]["battery_capacity"] = float(
            os.environ["NETRO_ROBOT_BATTERY_CAPACITY"]
        )

    # ROUTING - CVRP
    if "NETRO_CVRP_TIME_LIMIT" in os.environ:
        config["ROUTING"]["cvrp"]["time_limit_seconds"] = int(
            os.environ["NETRO_CVRP_TIME_LIMIT"]
        )

    # DEPLOYMENT
    if "NETRO_ROBOT_LAUNCH_TIME" in os.environ:
        config["DEPLOYMENT"]["robot_launch_time"] = float(
            os.environ["NETRO_ROBOT_LAUNCH_TIME"]
        )

    if "NETRO_ROBOT_RECOVERY_TIME" in os.environ:
        config["DEPLOYMENT"]["robot_recovery_time"] = float(
            os.environ["NETRO_ROBOT_RECOVERY_TIME"]
        )

    # I/O
    # I/O
    if "NETRO_DATASET_PATH" in os.environ:
        config["IO"]["dataset_path"] = os.environ["NETRO_DATASET_PATH"]

    if "NETRO_OUTPUT_PATH" in os.environ:
        config["IO"]["output_path"] = os.environ["NETRO_OUTPUT_PATH"]

    return config
