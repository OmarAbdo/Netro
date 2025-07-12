# Netro vs Traditional Truck-Only Delivery Comparison Report

## Key Metrics

| Metric | Traditional | Netro | Improvement |
|--------|------------|-------|-------------|
| Total Time (hours) | 721.07 | 16.63 (Hybrid: 16.63 + Last-Resort: 0.00) | 97.69% |
| Driver Cost (EUR) | 10816.04 | 4313.46 | 60.12% |
| Total Distance (km) | 7119.00 | 6686.04 | 6.08% |
| Total Cost | 18053.48 | 0.00 | 100.00% |
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
- Parallel hybrid operation time: 16.63 hours
- Truck travel time component (hybrid part): 86.09 hours
- Max cluster operation time (hybrid part): 13.44 hours
- Number of trucks (main hybrid): 34
- Number of clusters: 33
- **Total driver cost: 4313.46 EUR**

## Time Calculation Explanation

**Traditional System:**
- Total time: 721.1 hours (sum of all truck routes)
- Average per truck: 20.0 hours
- Driver cost: 721.1 hours × €15/h = €10816

**Netro System:**
- Hybrid parallel time: 16.6 hours (max across main truck routes to centroids + their cluster service time)
- Total Time: 16.6 hours
- Number of trucks (main hybrid): 34
- Total Driver Cost: €4313.46 (sum of all driver hours for hybrid and last-resort)

## Financial Analysis

The Netro system saves **6502.58 EUR** in driver costs,
representing a 60.12% reduction compared to the traditional approach.

This cost reduction is achieved through:
1. **Parallel truck operations**: Multiple trucks work simultaneously
2. **Parallel robot deliveries**: Robots deliver to customers simultaneously within clusters
3. **Strategic positioning**: Trucks only travel to cluster centroids
4. **Reduced total operation time**: From parallel operations

## Netro Additional Metrics

- Number of Clusters: 33
- Truck Distance: 5143.30 km
- Robot Distance: 1542.74 km
- Distance Ratio (Robot/Truck): 0.3:1