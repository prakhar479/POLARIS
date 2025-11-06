# Comprehensive Analysis Report: POLARIS vs NAIVE vs ADAMLS

This report provides a detailed comparison of the three adaptive approaches.

## Executive Summary

### Confidence Ranking:
1. **ADAMLS**: 0.7287
2. **NAIVE**: 0.6901
3. **POLARIS**: 0.6879

### Utility Ranking:
1. **NAIVE**: -1080.3311
2. **POLARIS**: -1433.4153
3. **ADAMLS**: -2324.1113

### Speed Ranking (Lower is Better):
1. **NAIVE**: 0.1882s
2. **POLARIS**: 0.1898s
3. **ADAMLS**: 0.2471s

## Detailed Analysis

### ADAMLS

- **Total Inferences**: 10,000
- **Duration**: 47.00 minutes
- **Inference Rate**: 212.78 inferences/minute
- **Models Used**: 5
- **Model Switches**: 865
- **Average Confidence**: 0.7287 ± 0.1362
- **Average Utility**: -2324.1113 ± 1735.3228
- **Average Processing Time**: 0.2471s ± 0.3088s

### NAIVE

- **Total Inferences**: 10,000
- **Duration**: 38.60 minutes
- **Inference Rate**: 259.04 inferences/minute
- **Models Used**: 4
- **Model Switches**: 36
- **Average Confidence**: 0.6901 ± 0.1591
- **Average Utility**: -1080.3311 ± 1045.7288
- **Average Processing Time**: 0.1882s ± 0.2854s

### POLARIS

- **Total Inferences**: 9,999
- **Duration**: 40.95 minutes
- **Inference Rate**: 244.19 inferences/minute
- **Models Used**: 5
- **Model Switches**: 112
- **Average Confidence**: 0.6879 ± 0.1578
- **Average Utility**: -1433.4153 ± 1366.0196
- **Average Processing Time**: 0.1898s ± 0.2552s

## Key Insights

- **Adaptivity**: Compare model switching frequency and patterns
- **Efficiency**: Analyze utility-to-time and utility-to-CPU ratios
- **Stability**: Examine variance in performance metrics
- **Scalability**: Consider inference rates and resource usage

## Visualizations

See the generated plots in individual approach directories and comparison plots in the main output directory.

