# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 363.91 | 236.97 (Hybrid: 26.80 + Last-Resort: 210.17) | 34.88% |
| Driver Cost (EUR) | 5458.61 | 4666.45 | 14.51% |
| Total Distance (km) | 3772.00 | 10823.66 | -186.95% |
| Total Cost | 9195.37 | 5693.35 | 38.08% |
| Number of Trucks | 18 | 20 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 363.91 hours
- Number of trucks: 18
- Average time per truck: 20.22 hours
- **Total driver cost: 5458.61 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 26.80 hours
- Last-resort truck time (sequential): 210.17 hours
- Combined Total Time: 236.97 hours
- Truck travel time component (hybrid part): 103.62 hours
- Max cluster operation time (hybrid part): 10.74 hours
- Number of trucks (main hybrid): 20
- Number of last-resort truck routes: 14
- Number of clusters: 20
- **Total driver cost: 4666.45 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 363.9 hours (sum of all truck routes)
- Average per truck: 20.2 hours
- Driver cost: 363.9 hours × €15/h = €5459

**Netro System:**
- Hybrid parallel time: 26.8 hours (max across main truck routes to centroids + their cluster service time)
- Last-resort truck time: 210.2 hours (sum of sequential last-resort truck routes)
- Combined Total Time: 237.0 hours
- Number of trucks (main hybrid): 20
- Total Driver Cost: €4666.45 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **792.16 EUR** in driver costs,
representing a 14.51% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

**Note about distance:** The total distance is higher for Netro because robots
must travel from cluster centroids to individual customers. However, the time
and cost savings from parallel operations more than compensate for this increase.

## Netro Additional Metrics

- Number of Clusters: 78
- Truck Distance: 9157.42 km
- Robot Distance: 1666.24 km
- Distance Ratio (Robot/Truck): 0.2:1