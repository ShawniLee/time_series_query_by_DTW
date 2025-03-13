# Time Series Query Matcher

## Project Overview

The Time Series Query Matcher is an efficient subsequence search system specifically designed for multidimensional time series data. It can quickly locate segments within a long time series (context sequence) that are similar to a given query sequence, even when these segments contain noise, scaling variations, or time shifts.

This project utilizes Dynamic Time Warping (DTW) algorithm to implement flexible similarity calculations and significantly improves search efficiency through various optimization techniques, including downsampling, early abandoning strategies, and LB_Keogh lower bound pruning.

## Visualization Examples

### Example Data Visualization
![Example Data Visualization](images/demo_example_data.png)

### Matching Results Visualization
![Matching Results Visualization](images/demo_matches.png)

### Performance Analysis Visualization
![Performance Analysis Visualization](images/demo_time_stats.png)

### Multi-Query Matching Visualization
![Multi-Query Matching Visualization](images/multi_query_demo_all_matches.png)

## Key Features

- **Multidimensional Time Series Support**: Can process multidimensional data simultaneously, suitable for complex application scenarios such as sensor fusion and motion recognition
- **Efficient Search Algorithms**:
  - Fast pruning using LB_Keogh lower bound to reduce DTW calculations
  - Sequence downsampling support to accelerate large-scale data processing
  - Early abandoning strategy implementation to avoid unnecessary complete calculations
- **Flexible Matching Capabilities**:
  - Support for similar pattern recognition with non-strict time alignment
  - Can process similar patterns with amplitude variations
  - Ability to identify patterns with noise interference
- **Multi-Query Support**:
  - Ability to search for multiple different patterns simultaneously in the same context sequence
  - Comprehensive visualization of multiple query results with color coding
  - Performance metrics for multi-query operations
- **Comprehensive Visualization Support**: Provides intuitive visualization of matching results and performance metrics
- **Multilingual Interface Support**: Automatic Chinese font adaptation, supporting Chinese display across different operating systems

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from time_series_matcher import TimeSeriesMatcher

# Create sample data
context = np.random.randn(10000, 2)  # 2D context sequence with length 10000
query = np.random.randn(100, 2)      # 2D query sequence with length 100

# Initialize the matcher
matcher = TimeSeriesMatcher(
    context, 
    threshold=0.5,         # DTW distance threshold
    radius=3,              # DTW warping radius
    position_group_ratio=0.1,  # Adjacent match grouping ratio
    lb_keogh_multiplier=1.5,   # LB_Keogh pruning multiplier
    downsample_factor=2        # Downsampling factor
)

# Find matches
matches, time_stats = matcher.find_matches(query)

# View matching results
print(f"Found {len(matches)} matches")
for pos, dist in matches[:5]:
    print(f"Position: {pos}, DTW distance: {dist:.4f}")

# Visualize matching results and save the image
matcher.visualize_matches(query, matches, save_path="images/my_matches.png")
```

### Multi-Query Usage

The project also supports searching for multiple different patterns simultaneously:

```python
from time_series_matcher import multi_query_demo

# Run the multi-query demo
multi_query_demo()
```

You can also implement your own multi-query search as follows:

```python
import numpy as np
from time_series_matcher import TimeSeriesMatcher, visualize_multi_query_matches

# Create context sequence
context = np.random.randn(10000, 2)

# Create multiple query sequences
query1 = np.random.randn(100, 2)  # First pattern
query2 = np.random.randn(80, 2)   # Second pattern
query3 = np.random.randn(120, 2)  # Third pattern

queries = [query1, query2, query3]
labels = ["Pattern A", "Pattern B", "Pattern C"]

# Initialize matcher with context sequence
matcher = TimeSeriesMatcher(
    context, 
    threshold=0.6,
    radius=2,
    position_group_ratio=0.1,
    lb_keogh_multiplier=1.2,
    downsample_factor=2
)

# Search for all patterns
all_matches = []
for query in queries:
    matches, _ = matcher.find_matches(query)
    all_matches.append(matches)

# Visualize all matches together
visualize_multi_query_matches(
    context,
    queries,
    all_matches,
    labels=labels,
    save_path="images/my_multi_query_matches.png"
)
```

### Built-in Demo

The project provides a complete demo function that generates synthetic data with specific patterns and displays matching results:

```python
from time_series_matcher import demo

# Run the single query demo
demo()
```

## Test Results

### Simple Sine Wave Pattern Matching
![Simple Sine Wave Pattern](images/test1_simple_pattern.png)

### Pattern Matching with Noise
![Pattern with Noise](images/test2_noisy_pattern.png)

### Pattern Matching with Different Scales
![Pattern with Different Scales](images/test3_scaled_pattern.png)

### Pattern Matching with Time Shifts
![Pattern with Time Shifts](images/test4_shifted_pattern.png)

### Multi-Query Pattern Matching
![Multi-Query Pattern Matching](images/multi_query_demo_all_matches.png)

## Implementation Principles

### 1. Core Algorithms

- **Dynamic Time Warping (DTW)**: Allows time series to undergo non-linear deformation along the time axis to find the optimal alignment
- **FastDTW**: An efficient approximate implementation of DTW with linear time and space complexity
- **LB_Keogh Lower Bound**: Quickly excludes impossible matching candidates by calculating the distance between the query sequence and the upper/lower envelopes of the context window

### 2. Optimization Strategies

- **Downsampling**: Reduces computation by downsampling data during the initial search phase
- **Early Abandoning**: Stops calculation once the cumulative distance exceeds the threshold while computing the LB_Keogh lower bound
- **Match Merging**: Combines matching results that are close in distance to avoid reporting multiple positions for the same pattern

### 3. Parameter Tuning

- **threshold**: Controls the strictness of matching, lower values demand more strict matching
- **radius**: DTW warping radius, affects the flexibility of time alignment
- **lb_keogh_multiplier**: Adjusts the strictness of LB_Keogh pruning
- **downsample_factor**: Controls the downsampling ratio, affecting the balance between computation speed and accuracy

## Application Scenarios

- **Anomaly Detection**: Identify segments in time series that are similar to known anomaly patterns
- **Pattern Discovery**: Find repeatedly occurring patterns in long time series
- **Gesture Recognition**: Match specific gesture patterns in sensor data
- **Biological Signal Analysis**: Identify specific waveforms in biological signals such as ECG and EEG
- **Multi-pattern Search**: Simultaneously search for multiple different patterns in the same data stream

## Performance Evaluation

The project provides performance evaluation and visualization tools to analyze the time proportion of each stage:
- Preprocessing stage
- LB_Keogh pruning stage
- DTW calculation stage
- Post-processing stage

## Test Suite

By running `test_matcher.py`, you can verify the algorithm's performance in different scenarios:
- Simple sine wave pattern matching
- Pattern matching with noise
- Pattern matching with different scales
- Pattern matching with time shifts 