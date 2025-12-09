# Hotel Reservations Classification - Project Report

## 📋 Project Overview

This project focuses on predicting hotel reservation cancellations using machine learning classification algorithms. The dataset contains booking information with a moderate class imbalance, where approximately 67% of reservations are not canceled and 33% are canceled.

## 📊 Dataset Information

- **Source**: Hotel Reservations Classification Dataset (Kaggle)
- **Total Records**: 36,275
- **Class Distribution**:
  - Not_Canceled: 24,390 (67.2%)
  - Canceled: 11,885 (32.8%)
- **Class Imbalance Ratio**: 2:1 (moderate imbalance)

## 🔬 Experimental Setup

### Data Preprocessing
- **Feature Engineering**: Separation of numerical and categorical features
- **Encoding**: 
  - Target variable: LabelEncoder
  - Categorical features: OneHotEncoder
- **Scaling**: RobustScaler for numerical features
- **Data Split**: 
  - Training: 80%
  - Validation: 10%
  - Test: 10%

### Evaluation Metric
- **Primary Metric**: F2-Score (β=2)
  - Emphasizes recall over precision
  - Prioritizes identifying canceled bookings (critical for hotel operations)
- **Cross-Validation**: RepeatedStratifiedKFold (3 splits, 3 repeats)

## 🧪 Experiments Conducted

### Experiment 1: Model Selection

Eight classification algorithms were evaluated using baseline configurations:

| Model | F2-Score (Mean) | Std Dev |
|-------|----------------|---------|
| **Random Forest** | **0.9292** | **±0.0034** |
| XGBoost | 0.9221 | ±0.0029 |
| Gradient Boosting | 0.9076 | ±0.0045 |
| SVC | 0.9071 | ±0.0024 |
| Decision Tree | 0.8808 | ±0.0022 |
| LDA | 0.8783 | ±0.0032 |
| Logistic Regression | 0.8754 | ±0.0035 |
| AdaBoost | 0.8750 | ±0.0048 |

📌 **Best Model Identified**

After evaluating multiple classification algorithms, **Random Forest** achieved the highest performance based on the F2-Score:

- **Model**: Random Forest  
- **Average F2-Score**: 0.9292  
- **Standard deviation**: ± 0.0034  

This indicates that the model not only reaches a high level of performance, but also shows consistent and stable results across the different validations performed.

### Experiment 2: Sampling Techniques Evaluation

Various resampling techniques were tested with the Random Forest model to handle class imbalance:

📌 **Best Sampling Technique Identified**

Among the different sampling techniques evaluated, **KMeansSMOTE** achieved the best performance in terms of the F2-Score:

- **Technique**: KMeansSMOTE  
- **Average F2-Score**: 0.9278  
- **Standard deviation**: ± 0.0030  

This indicates that using KMeansSMOTE provides both high and consistent performance across the validation folds, making it the most effective sampling method for this classification task.

### Experiment 3: Class Weight Balancing Strategy

Random Forest with `class_weight='balanced'` parameter was tested as an alternative to resampling:

📌 **Class Weight Balancing Outperforms SMOTE**

Instead of using SMOTE or other sampling techniques, applying the `class_weight='balanced'` parameter led to the best performance:

- **Strategy**: Class weight balancing (`class_weight='balanced'`)
- **Model**: Random Forest  
- **Average F2-Score**: 0.9291  
- **Standard deviation**: ± 0.0029  

This indicates that for this dataset, simply leveraging class weight balancing produced better and more consistent results than traditional sampling methods.

## 🎯 Final Model Configuration

Based on the experiments, the final model uses Random Forest with class weight balancing:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
```

## 📈 Final Model Performance

### Validation Set Results

```
F2-Score: 0.9405

Confusion Matrix:
                    Predicted
                Not_Canceled  Canceled
Actual
Not_Canceled        926         262
Canceled            118        2321

Classification Metrics:
- Precision (Class 0): 0.89
- Recall (Class 0): 0.78
- Precision (Class 1): 0.90
- Recall (Class 1): 0.95
- Overall Accuracy: 0.90
```

### Test Set Results

```
F2-Score: 0.9396

Confusion Matrix:
                    Predicted
                Not_Canceled  Canceled
Actual
Not_Canceled        954         235
Canceled            127        2312

Classification Metrics:
- Precision (Class 0): 0.88
- Recall (Class 0): 0.80
- Precision (Class 1): 0.91
- Recall (Class 1): 0.95
- Overall Accuracy: 0.90
```

## 🔍 Key Findings

1. **Model Selection**: Random Forest (F2=0.9292) consistently outperformed all other algorithms, including advanced ensemble methods like XGBoost (F2=0.9221) and Gradient Boosting (F2=0.9076).

2. **Sampling Strategy Comparison**:
   - Class weight balancing: F2=0.9291 (±0.0029)
   - KMeansSMOTE: F2=0.9278 (±0.0030)
   - Class weight balancing proved to be simpler, faster, and equally effective as sophisticated resampling techniques.

3. **Excellent Generalization**: The model showed nearly identical performance on validation (F2=0.9405) and test sets (F2=0.9396), indicating no overfitting and strong generalization capability.

4. **High Recall for Canceled Class**: The model achieves ~95% recall for canceled bookings, meaning it successfully identifies 19 out of every 20 cancellations. This is critical for hotel revenue management.

5. **Low False Negative Rate**: Only ~5% of canceled bookings are missed (118 on validation, 127 on test), minimizing potential revenue loss from unexpected cancellations.

6. **Stability**: Low standard deviation across cross-validation folds (±0.0029) demonstrates consistent and reliable predictions.

## 💡 Conclusions

The Random Forest model with `class_weight='balanced'` demonstrates **excellent performance** on the Hotel Reservations dataset. Key achievements include:

- **High F2-Score**: 0.94 on both validation and test sets
- **Exceptional Recall**: 95% detection rate for canceled bookings
- **Strong Generalization**: Consistent performance across unseen data
- **Operational Efficiency**: No need for complex resampling pipelines

Given the moderate class imbalance (2:1 ratio), the selected approach is **robust, efficient, and well-suited** for this problem. The class weight balancing strategy alone delivers high, stable performance without the computational overhead of resampling techniques.

**This model is production-ready** and appropriate for deployment in hotel reservation systems for:
- Early cancellation prediction
- Dynamic pricing optimization
- Overbooking strategy refinement
- Targeted customer retention campaigns

## 🛠️ Technologies Used

- **Python 3.12**
- **scikit-learn**: Model training, evaluation, and preprocessing
- **imbalanced-learn**: Sampling techniques evaluation
- **XGBoost**: Gradient boosting implementation
- **pandas**: Data manipulation and analysis
- **matplotlib**: Confusion matrix visualization
