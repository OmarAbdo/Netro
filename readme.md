# Netro: Last-Mile Delivery Optimization with Autonomous Delivery Robots

Netro is a state-of-the-art system for optimizing the last-mile delivery problem through a hybrid approach combining trucks and autonomous delivery robots. This project is designed for researchers and practitioners in logistics who are exploring innovative solutions to reduce costs and improve efficiency in the last-mile delivery segment.

## Key Features

- **Hybrid Delivery System**: Innovative truck-and-robot approach where trucks serve as mobile depots for robot delivery
- **Advanced Clustering**: Density-based customer clustering with capacity-aware splitting
- **Metaheuristic Optimization**: Implements both exact methods and metaheuristics for routing optimization
- **Comprehensive Benchmarking**: Compares the hybrid approach against traditional truck-only delivery
- **High-Quality Implementation**: Clean code architecture following SOLID principles
- **Realistic Modeling**: Accounts for real-world constraints like battery capacity, time windows, and outlier handling

## Project Structure

```
netro/
├── config/               # Configuration settings
│   ├── default.py        # Default configuration values
│   └── env.py            # Environment-specific configuration
├── core/                 # Core domain entities and interfaces
│   ├── entities/         # Domain objects (Location, Vehicle, Cluster, etc.)
│   └── interfaces/       # Abstractions for algorithms
├── services/             # Application services
│   ├── clustering/       # Customer clustering algorithms
│   ├── routing/          # Routing algorithms for trucks and robots
│   ├── benchmarking/     # Comparison of approaches
│   └── robot_deployment/ # Robot loading and deployment strategies
├── infrastructure/       # External adapters
│   ├── io/               # Input/output handling
│   └── visualization/    # Result visualization
├── application.py        # Application orchestration logic
└── main.py               # Command-line interface entry point
```

## Mathematical Formulation

The traditional truck-only delivery approach can be modeled as:

**T(traditional) = N × (t(travel-customer) + t(service))**

Where:

- N: number of customers
- t(travel-customer): average travel time between customers
- t(service): service time per customer

This leads to linear growth with O(N) complexity.

The Netro hybrid approach follows:

**T(Netro) = K × (t(travel-cluster) + t(unloading)) + [M/R × (t(travel-robot-customer) + t(robot-service)) + t(recovery)]**

Where:

- K: number of clusters
- M: customers per cluster
- R: robots per cluster
- t(travel-cluster): travel time between cluster centroids
- t(unloading): time to unload and prepare robots
- t(travel-robot-customer): average robot travel time
- t(robot-service): robot service time per customer
- t(recovery): robot recovery time

This formula demonstrates sub-linear scaling and improved efficiency under the right conditions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/netro.git
cd netro
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the full Netro workflow with a dataset:

```bash
python -m netro.main --dataset c101.txt --robot starship
```

This will:

1. Load the specified dataset
2. Perform customer clustering
3. Run the baseline truck-only solution
4. Run the Netro hybrid solution
5. Compare and visualize the results

### Configuration

You can configure Netro by:

1. Editing `netro/config/default.py` for permanent changes
2. Setting environment variables for temporary overrides
3. Creating a `.env` file for project-specific settings

Key configuration options:

```
# Clustering
NETRO_MIN_CLUSTER_SIZE=5
NETRO_CLUSTERING_ALGORITHM=hdbscan

# Vehicles
NETRO_TRUCK_CAPACITY=200
NETRO_TRUCK_SPEED=60
NETRO_TRUCK_ROBOT_CAPACITY=5
NETRO_ROBOT_CAPACITY=30
NETRO_ROBOT_SPEED=15
NETRO_ROBOT_BATTERY_CAPACITY=120

# Paths
NETRO_DATASET_PATH=dataset/
NETRO_OUTPUT_PATH=output/
```

## Datasets

The project uses the Solomon VRPTW benchmark instances, which are standard in the vehicle routing literature. These datasets include:

- Customer coordinates
- Demand values
- Time windows
- Service times

The datasets are not included in the repository but can be downloaded from:
http://web.cba.neu.edu/~msolomon/problems.htm

## Extending the Project

### Adding New Clustering Algorithms

1. Implement the `ClusteringAlgorithm` protocol in `core/interfaces/clustering.py`
2. Add your implementation to the `services/clustering/` directory
3. Update the configuration to use your algorithm

### Adding New Routing Algorithms

1. Implement the `RoutingAlgorithm` protocol in `core/interfaces/routing.py`
2. Add your implementation to the `services/routing/` directory
3. Update the configuration to use your algorithm

## Research Foundation

This project builds upon several academic works in the field of last-mile delivery optimization:

- Chen, S., et al. (2018), "An adaptive large neighborhood search heuristic for dynamic vehicle routing problems", Computers and Electrical Engineering, 67, 596–607.
- Simoni et al. (2020), "Optimization and analysis of a robot-assisted last mile delivery system", Transportation Research Part E.
- Mourelo Ferrandez et al. (2016), "Optimization of a Truck-drone in Tandem Delivery Network Using K-means and Genetic Algorithm", Journal of Industrial Engineering and Management.
- Ostermeier et al. (2022), "Cost-optimal truck-and-robot routing for last-mile delivery".

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Netro in your research, please cite:

```
@mastersthesis{abdou2025netro,
  author  = {Abdou, Omar},
  title   = {Netro: Methods and Applications of Autonomous Delivery Robots for Last-Mile Delivery Optimization},
  school  = {European University Viadrina},
  year    = {2025},
  address = {Frankfurt (Oder), Germany}
}
```
