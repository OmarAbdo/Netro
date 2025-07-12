# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 721.07 | 472.65 (Hybrid: 15.45 + Last-Resort: 457.20) | 34.45% |
| Sequential Time Equivalent | - | 268.65 | - |
| Time Savings from Parallelization | - | 253.20h (94.3%) | - |
| Driver Cost (EUR) | 10816.04 | 10887.75 | -0.66% |
| Total Distance (km) | 7119.00 | 12583.01 | -76.75% |
| Number of Trucks | 36 | 34 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 721.07 hours
- Number of trucks: 36
- Average time per truck: 20.03 hours
- **Total driver cost: 10816.04 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 15.45 hours
- Last-resort truck time (sequential): 457.20 hours
- Combined Total Time: 472.65 hours
- Truck travel time component (hybrid part): 86.09 hours
- Max cluster operation time (hybrid part): 12.32 hours
- Number of trucks (main hybrid): 34
- Number of last-resort truck routes: 29
- Number of clusters: 33
- **Total driver cost: 10887.75 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 721.1 hours (sum of all truck routes)
- Average per truck: 20.0 hours
- Driver cost: 721.1 hours × €15/h = €10816

**Netro System:**
- Hybrid parallel time: 15.4 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 457.2 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 472.7 hours
- Number of trucks (main hybrid): 34
- Total Driver Cost: €10887.75 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **-71.71 EUR** in driver costs,
representing a -0.66% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 33
- Truck Distance: 10887.30 km
- Robot Distance: 1695.71 km
- Distance Ratio (Robot/Truck): 0.2:1