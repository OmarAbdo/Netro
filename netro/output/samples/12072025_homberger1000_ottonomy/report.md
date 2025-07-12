# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 2190.79 | 20.22 (Hybrid: 20.22 + Last-Resort: 0.00) | 99.08% |
| Sequential Time Equivalent | - | 498.18 | - |
| Time Savings from Parallelization | - | 477.95h (95.9%) | - |
| Driver Cost (EUR) | 32861.87 | 7472.63 | 77.26% |
| Total Distance (km) | 41071.00 | 40293.86 | 1.89% |
| Number of Trucks | 90 | 85 | - |

## Operational Analysis

### Traditional Approach
- Operation mode: Sequential customer visits (time is sum across all trucks)
- Total operation time: 2190.79 hours
- Number of trucks: 90
- Average time per truck: 24.34 hours
- **Total driver cost: 32861.87 EUR**

### Netro Hybrid Approach
- Operation mode: Parallel trucks to centroids, then parallel robots; sequential last-resort trucks if needed
- Parallel hybrid operation time: 20.22 hours
- Truck travel time component (hybrid part): 582.03 hours
- Max cluster operation time (hybrid part): 12.14 hours
- Number of trucks (main hybrid): 85
- Number of clusters: 84
- **Total driver cost: 7472.63 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 2190.8 hours (sum of all truck routes)
- Average per truck: 24.3 hours
- Driver cost: 2190.8 hours × €15/h = €32862

**Netro System:**
- Hybrid parallel time: 20.2 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 20.2 hours
- Number of trucks (main hybrid): 85
- Total Driver Cost: €7472.63 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **25389.24 EUR** in driver costs,
representing a 77.26% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

## Netro Additional Metrics

- Number of Clusters: 84
- Truck Distance: 34857.56 km
- Robot Distance: 5436.29 km
- Distance Ratio (Robot/Truck): 0.2:1