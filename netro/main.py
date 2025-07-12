# netro/main.py
import argparse
import sys
from netro.application import NetroApplication
from netro.config.env import get_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Netro - Last-Mile Delivery Optimization with Autonomous Robots"
    )

    # Dynamically get robot choices from config
    config = get_config()
    robot_choices = list(config["VEHICLES"]["robots"].keys())

    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset file name in the dataset directory (e.g., c101.txt)",
    )

    parser.add_argument(
        "--robot",
        "-r",
        required=True,
        choices=robot_choices,
        help="Specify the type of robot to use for the simulation.",
    )

    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization saving"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Initialize application with the selected robot
        app = NetroApplication(robot_type=args.robot)

        # Run workflow
        result = app.run_full_workflow(
            dataset_name=args.dataset, save_visualizations=not args.no_viz
        )

        print("\nWorkflow completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
