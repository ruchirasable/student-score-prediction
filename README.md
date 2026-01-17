# Student Score Prediction — ML Regression Project

This project predicts student exam scores based on study hours using a supervised Machine Learning model (Linear Regression).

---

## 📂 Dataset

- File: `student_scores.csv`
- Features:
  - `Hours` → number of hours studied
- Target:
  - `Scores` → exam score out of 100
- Dataset Size: 25 rows
- Source: Public sample dataset (common for regression demos)

---

## 🧠 Model Choice

**Linear Regression** was chosen because:
- Relationship between hours and score is approximately linear
- Model is simple, interpretable, and performs well on small numeric datasets

Model file: `model.py`

---

## 🏗 Training Process

Steps executed in `model.py`:

1. Load CSV dataset using Pandas
2. Split dataset into train/test sets (80/20)
3. Fit Linear Regression model on training data
4. Predict on test set
5. Evaluate performance metrics

---

## 📊 Metrics

Evaluation metrics used:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

Sample output (ranges based on dataset):
- MAE: ~4.5
- MSE: ~25
- R²: ~0.93

This shows the model generalizes well for such a small dataset.

---

## 🧪 Experiments Tried

### 1. **Without Train-Test Split**
- Model trained on full dataset
- Produced optimistic results (overfitting risk)

### 2. **With 80/20 Train-Test Split**
- More realistic performance estimate
- Used for reported metrics

## 3.Removing Outliers**
- Tested removing a high-hours sample
- Slightly reduced MSE, improved R²
- Not applied to final results due to small dataset size

---

## 🐞 Observed Errors / Limitations

- Small dataset → model performance sensitive to split
- Scores do not perfectly scale linearly (noise present)
- Outliers can strongly influence slope in Linear Regression

---

## ▶️ Inference (Prediction)

Example usage:

```python
input_hours = [[5]]  # 5 hours studied
predicted_score = model.predict(input_hours)
print(predicted_score)

