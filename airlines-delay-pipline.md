
# Airlines Data Pipeline Documentation

## Pipeline Overview
This pipeline preprocesses airline-related data for machine learning. It includes target encoding for categorical features, outlier detection and treatment using Tukey fences, robust feature scaling, and missing value imputation using KNN.

<img width="639" alt="Screenshot 2025-06-10 at 1 57 01 PM" src="https://github.com/user-attachments/assets/679a80a4-4f91-42ed-8529-a5bb892396c8" />


## Step-by-Step Design Choices

### 1. Target Encoding for Airline (`target_airline`)
- **Transformer:** `CustomTargetTransformer(col='Airline', smoothing=10)`
- **Design Choice:** Target encoding with smoothing factor of 10
- **Rationale:**
  - Encodes 'Airline' based on its relationship with the target variable
  - Reduces overfitting caused by rare airlines through smoothing

### 2. Target Encoding for AirportFrom (`target_airportfrom`)
- **Transformer:** `CustomTargetTransformer(col='AirportFrom', smoothing=10)`
- **Design Choice:** Target encoding with smoothing
- **Rationale:**
  - Captures average target value for each origin airport
  - Smoothed to account for airports with few data points

### 3. Target Encoding for AirportTo (`target_airportto`)
- **Transformer:** `CustomTargetTransformer(col='AirportTo', smoothing=10)`
- **Design Choice:** Target encoding with smoothing
- **Rationale:**
  - Encodes destination airport using target-based mean encoding
  - Smoothed to reduce variance for underrepresented destinations

### 4. Outlier Treatment for Time (`tukey_time`)
- **Transformer:** `CustomTukeyTransformer(target_column='Time', fence='outer')`
- **Design Choice:** Tukey method with outer fence
- **Rationale:**
  - Removes extreme values for flight time
  - Outer fence (Q1–3×IQR, Q3+3×IQR) preserves valid but high-variance records

### 5. Outlier Treatment for Length (`tukey_length`)
- **Transformer:** `CustomTukeyTransformer(target_column='Length', fence='outer')`
- **Design Choice:** Tukey outer fence for outlier detection
- **Rationale:**
  - Handles rare anomalies in flight length
  - Retains most of the original distribution by targeting only extreme outliers

### 6. Scaling of Time (`scale_time`)
- **Transformer:** `CustomRobustTransformer(target_column='Time')`
- **Design Choice:** RobustScaler based on median and IQR
- **Rationale:**
  - Rescales Time to reduce outlier influence
  - More appropriate than StandardScaler for skewed flight durations

### 7. Scaling of Length (`scale_length`)
- **Transformer:** `CustomRobustTransformer(target_column='Length')`
- **Design Choice:** Robust scaling
- **Rationale:**
  - Normalizes Length feature for model input
  - Minimizes outlier impact on scaling process

### 8. Target Encoding for DayOfWeek (`target_dayofweek`)
- **Transformer:** `CustomTargetTransformer(col='DayOfWeek', smoothing=10)`
- **Design Choice:** Target encoding for day of the week
- **Rationale:**
  - Captures flight outcome patterns across different weekdays
  - Smoothed to prevent overfitting for less frequent days

### 9. Imputation (`impute`)
- **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
- **Design Choice:** KNN imputation with k=5
- **Rationale:**
  - Fills missing values using patterns in the dataset
  - KNN provides more informed estimates than mean/median by leveraging feature proximity

## Pipeline Execution Order Rationale
1. **Target encoding** is performed early since it relies on original categorical values.
2. **Outlier treatment** is applied before scaling to prevent skewing scale parameters.
3. **Robust scaling** is done after removing outliers to normalize features.
4. **Imputation** is performed last to fill in missing values using the most refined features.

## Performance Considerations
- **Target Encoding:** Smoothing reduces overfitting from high-cardinality features like airport codes.
- **Tukey Outlier Removal:** Conservative outlier handling ensures minimal data loss while improving model robustness.
- **Robust Scaling:** Median and IQR scaling is resilient to remaining anomalies.
- **KNN Imputation:** Leverages correlations between features to estimate missing values accurately.
