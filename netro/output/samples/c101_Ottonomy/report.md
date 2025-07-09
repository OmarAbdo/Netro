# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 363.91 | 64.55 (Hybrid: 64.55 + Last-Resort: 0.00) | 82.26% |
| Driver Cost (EUR) | 5458.61 | 3920.75 | 28.17% |
| Total Distance (km) | 3772.00 | 4937.38 | -30.90% |
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
- Parallel hybrid operation time: 64.55 hours
- Truck travel time component (hybrid part): 37.37 hours
- Max cluster operation time (hybrid part): 60.24 hours
- Number of trucks (main hybrid): 15
- Number of clusters: 15
- **Total driver cost: 3920.75 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 363.9 hours (sum of all truck routes)
- Average per truck: 20.2 hours
- Driver cost: 363.9 hours × €15/h = €5459

**Netro System:**
- Hybrid parallel time: 64.6 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 64.6 hours
- Number of trucks (main hybrid): 15
- Total Driver Cost: €3920.75 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **1537.86 EUR** in driver costs,
representing a 28.17% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 16
- Truck Distance: 2242.49 km
- Robot Distance: 2694.89 km
- Distance Ratio (Robot/Truck): 1.2:1