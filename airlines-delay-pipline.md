Airlines Data Pipeline Documentation
Pipeline Overview
This pipeline preprocesses the Airlines dataset to prepare it for machine learning modeling. It handles high-cardinality categorical encoding through target encoding, feature scaling for numerical variables, and missing value imputation to predict flight delays.



<img width="663" alt="Screenshot 2025-06-10 at 10 58 04 AM" src="https://github.com/user-attachments/assets/7c8a63ec-1f3b-432e-ab62-49e3c9fdbbe3" />




Step-by-Step Design Choices
1. Airline Target Encoding (target_airline)

Transformer: CustomTargetTransformer(col='Airline', smoothing=10)
Design Choice: Target encoding with smoothing factor of 10 for airline carriers
Rationale:

Airlines have different operational patterns affecting delay rates
Target encoding captures each airline's historical delay performance
Smoothing=10 prevents overfitting to rare airline codes while preserving signal



2. Departure Airport Target Encoding (target_airportfrom)

Transformer: CustomTargetTransformer(col='AirportFrom', smoothing=10)
Design Choice: Target encoding with smoothing factor of 10 for departure airports
Rationale:

High cardinality feature (hundreds of airports) makes one-hot encoding impractical
Different airports have varying delay patterns due to weather, traffic, infrastructure
Smoothing handles airports with few flights by blending with global delay rate



3. Arrival Airport Target Encoding (target_airportto)

Transformer: CustomTargetTransformer(col='AirportTo', smoothing=10)
Design Choice: Target encoding with smoothing factor of 10 for destination airports
Rationale:

Destination airports influence arrival delays due to congestion and capacity
Target encoding efficiently handles high cardinality while preserving delay relationships
Consistent smoothing across airport features ensures balanced regularization



4. Time Scaling (scale_time)

Transformer: CustomRobustTransformer(target_column='Time')
Design Choice: Robust scaling for departure time feature
Rationale:

Departure times may have outliers (early morning or late night flights)
Robust scaling uses median and IQR, less sensitive to extreme departure times
Preserves time-of-day patterns while normalizing scale for ML algorithms



5. Flight Length Scaling (scale_length)

Transformer: CustomRobustTransformer(target_column='Length')
Design Choice: Robust scaling for flight duration feature
Rationale:

Flight durations vary widely (short regional to long transcontinental flights)
Robust scaling handles the natural skewness in flight length distribution
Prevents longer flights from dominating distance calculations in ML models



6. Day of Week Target Encoding (target_dayofweek)

Transformer: CustomTargetTransformer(col='DayOfWeek', smoothing=10)
Design Choice: Target encoding with smoothing factor of 10 for day of week
Rationale:

Different days have varying delay patterns (business vs. leisure travel)
Target encoding captures day-specific delay rates better than ordinal encoding
Smoothing ensures stable estimates across all days of the week



7. Imputation (impute)

Transformer: CustomKNNTransformer(n_neighbors=5)
Design Choice: KNN imputation with 5 neighbors
Rationale:

Uses relationships between features to estimate missing values intelligently
k=5 balances between using enough context and avoiding noise
More sophisticated than mean/median imputation for mixed feature types



Pipeline Execution Order Rationale

Target encoding first for all categorical features while preserving original relationships
Scaling applied to numerical features after categorical transformations complete
Day of week target encoding positioned after other encodings for consistency
Imputation last to handle any missing values using all preprocessed features

Performance Considerations

Target encoding instead of one-hot encoding reduces dimensionality significantly
Robust scaling chosen over standard scaling due to potential outliers in time/duration
KNN imputation preserves feature relationships better than simple imputation methods
Consistent smoothing factor (10) across target encoders for balanced regularization
