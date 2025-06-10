# Airlines Data Pipeline Documentation

## Pipeline Overview

This pipeline preprocesses the Airlines dataset to prepare it for machine learning modeling. It handles:

- High-cardinality categorical encoding through **target encoding**
- **Feature scaling** for numerical variables
- **Missing value imputation** to predict flight delays

![Airlines Pipeline](https://github.com/user-attachments/assets/7c8a63ec-1f3b-432e-ab62-49e3c9fdbbe3)

---

## Step-by-Step Design Choices

### 1. Airline Target Encoding (`target_airline`)

- **Transformer**: `CustomTargetTransformer(col='Airline', smoothing=10)`
- **Design Choice**: Target encoding with smoothing factor of 10 for airline carriers

**Rationale**:
- Airlines have different operational patterns affecting delay rates  
- Target encoding captures each airline's historical delay performance  
- Smoothing prevents overfitting to rare airline codes

---

### 2. Departure Airport Target Encoding (`target_airportfrom`)

- **Transformer**: `CustomTargetTransformer(col='AirportFrom', smoothing=10)`
- **Design Choice**: Target encoding for departure airports

**Rationale**:
- High cardinality makes one-hot encoding impractical  
- Airports have varying delay patterns due to weather, traffic, etc.  
- Smoothing blends rare values with the global delay rate

---

### 3. Arrival Airport Target Encoding (`target_airportto`)

- **Transformer**: `CustomTargetTransformer(col='AirportTo', smoothing=10)`
- **Design Choice**: Target encoding for destination airports

**Rationale**:
- Destination airports influence arrival delays  
- Encoding preserves high-cardinality relationships  
- Same smoothing factor ensures balanced regularization

---

### 4. Time Scaling (`scale_time`)

- **Transformer**: `CustomRobustTransformer(target_column='Time')`
- **Design Choice**: Robust scaling for departure time

**Rationale**:
- Departure times may include outliers  
- Robust scaling (median + IQR) is less sensitive to extremes  
- Normalizes time-of-day patterns for ML

---

### 5. Flight Length Scaling (`scale_length`)

- **Transformer**: `CustomRobustTransformer(target_column='Length')`
- **Design Choice**: Robust scaling for flight duration

**Rationale**:
- Flight durations have skewed distribution  
- Scaling prevents long flights from dominating models  
- Maintains meaningful distance relationships

---

### 6. Day of Week Target Encoding (`target_dayofweek`)

- **Transformer**: `CustomTargetTransformer(col='DayOfWeek', smoothing=10)`
- **Design Choice**: Target encoding for day of the week

**Rationale**:
- Different days show distinct delay patterns  
- More informative than ordinal encoding  
- Smoothing ensures stability across all weekdays

---

### 7. Imputation (`impute`)

- **Transformer**: `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice**: KNN imputation with 5 neighbors

**Rationale**:
- Learns from feature relationships  
- `k=5` balances context and noise  
- More intelligent than simple mean/median filling

---

## Pipeline Execution Order Rationale

1. **Target encoding** for categorical features first to retain relationships  
2. **Scaling** applied to numerical features afterward  
3. **Day of week** encoded after airports and airlines  
4. **Imputation** comes last using fully preprocessed features

---

## Performance Considerations

- Target encoding drastically reduces dimensionality  
- Robust scaling handles outliers better than standard scaling  
- KNN imputation maintains feature correlations  
- Consistent smoothing (`10`) improves generalization across encodings
