# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 164.14 | 93.19 (Hybrid: 10.20 + Last-Resort: 82.99) | 43.23% |
| Sequential Time Equivalent | - | 78.26 | - |
| Time Savings from Parallelization | - | 68.06h (87.0%) | - |
| Driver Cost (EUR) | 2462.16 | 2418.75 | 1.76% |
| Total Distance (km) | 818.00 | 1796.10 | -119.57% |
| Number of Trucks | 10 | 11 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 164.14 hours
- Number of trucks: 10
- Average time per truck: 16.41 hours
- **Total driver cost: 2462.16 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 10.20 hours
- Last-resort truck time (sequential): 82.99 hours
- Combined Total Time: 93.19 hours
- Truck travel time component (hybrid part): 11.81 hours
- Max cluster operation time (hybrid part): 8.44 hours
- Number of trucks (main hybrid): 11
- Number of last-resort truck routes: 7
- Number of clusters: 10
- **Total driver cost: 2418.75 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 164.1 hours (sum of all truck routes)
- Average per truck: 16.4 hours
- Driver cost: 164.1 hours × €15/h = €2462

**Netro System:**
- Hybrid parallel time: 10.2 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 83.0 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 93.2 hours
- Number of trucks (main hybrid): 11
- Total Driver Cost: €2418.75 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **43.40 EUR** in driver costs,
representing a 1.76% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 10
- Truck Distance: 1344.47 km
- Robot Distance: 451.63 km
- Distance Ratio (Robot/Truck): 0.3:1