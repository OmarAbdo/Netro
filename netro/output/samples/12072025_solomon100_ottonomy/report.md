# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 164.14 | 8.58 (Hybrid: 8.58 + Last-Resort: 0.00) | 94.77% |
| Driver Cost (EUR) | 2462.16 | 842.42 | 65.79% |
| Total Distance (km) | 818.00 | 1153.22 | -40.98% |
| Total Cost | 3707.18 | 0.00 | 100.00% |
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
- Parallel hybrid operation time: 8.58 hours
- Truck travel time component (hybrid part): 11.81 hours
- Max cluster operation time (hybrid part): 6.82 hours
- Number of trucks (main hybrid): 11
- Number of clusters: 10
- **Total driver cost: 842.42 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 164.1 hours (sum of all truck routes)
- Average per truck: 16.4 hours
- Driver cost: 164.1 hours × €15/h = €2462

**Netro System:**
- Hybrid parallel time: 8.6 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 8.6 hours
- Number of trucks (main hybrid): 11
- Total Driver Cost: €842.42 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1619.74 EUR** in driver costs,
representing a 65.79% reduction compared to the traditional approach.

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
- Truck Distance: 699.47 km
- Robot Distance: 453.75 km
- Distance Ratio (Robot/Truck): 0.6:1