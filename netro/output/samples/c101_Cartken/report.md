# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 363.91 | 91.50 (Hybrid: 91.50 + Last-Resort: 0.00) | 74.86% |
| Driver Cost (EUR) | 5458.61 | 5307.79 | 2.76% |
| Total Distance (km) | 3772.00 | 3809.75 | -1.00% |
| Total Cost | 9195.37 | 0.00 | 100.00% |
| Number of Trucks | 18 | 15 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 363.91 hours
- Number of trucks: 18
- Average time per truck: 20.22 hours
- **Total driver cost: 5458.61 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 91.50 hours
- Truck travel time component (hybrid part): 31.98 hours
- Max cluster operation time (hybrid part): 87.19 hours
- Number of trucks (main hybrid): 15
- Number of clusters: 15
- **Total driver cost: 5307.79 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 363.9 hours (sum of all truck routes)
- Average per truck: 20.2 hours
- Driver cost: 363.9 hours × €15/h = €5459

**Netro System:**
- Hybrid parallel time: 91.5 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 91.5 hours
- Number of trucks (main hybrid): 15
- Total Driver Cost: €5307.79 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **150.82 EUR** in driver costs,
representing a 2.76% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 15
- Truck Distance: 1918.74 km
- Robot Distance: 1891.01 km
- Distance Ratio (Robot/Truck): 1.0:1