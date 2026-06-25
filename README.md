# Task 5 - The First Prediction
**PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion**

## Results
| Metric | Value |
|---|---|
| Baseline Accuracy | 0.3000 |
| Model Accuracy (Validation) | 0.9333 |
| Improvement | +0.6333 |
| Wrong Predictions | 2 / 30 |

## Model
- Algorithm: DecisionTreeClassifier(max_depth=3)
- Dataset: Iris (150 samples, 3 classes)
- Split: 60% Train / 20% Validation / 20% Test

## Error Notes
Both misclassifications confused versicolor to virginica due to overlapping petal measurements.

## Stack
- Python 3.13
- scikit-learn
- pandas
- numpy
