# 🧠 Day 09: Naive Bayes Classification

A practical guide to implementing Naive Bayes classifiers for multi-class and binary classification using scikit-learn.

---

## 📚 Two Implementation Files

### 1️⃣ **NaiveBayes.ipynb** - Multi-class Naive Bayes

**🎯 What it does:**
Implements Gaussian Naive Bayes for 3-class classification using synthetic dataset with performance metrics.

**🔑 Key Steps:**

```
📥 Generate Data (800 samples, 6 features, 3 classes)
     ↓
🎨 Visualize: Scatter plot of features
     ↓
✂️ Split Data: 67% train, 33% test
     ↓
🤖 Build Model: GaussianNB classifier
     ↓
🎓 Train: Fit on training data
     ↓
🎯 Predict: Classify test samples
     ↓
📊 Evaluate: Accuracy & F1 score
     ↓
📈 Confusion Matrix: Detailed performance
```

---

### 2️⃣ **NaiveBayesEx.ipynb** - Binary Naive Bayes (Real Data)

**🎯 What it does:**
Applies Naive Bayes to real-world Advertising dataset for sales classification (High/Low).

**🔑 Key Steps:**

```
📥 Load Data: Advertising.csv
     ↓
🔢 Convert: Sales column to numeric
     ↓
🏷️ Classify: High (>15) vs Low (≤15)
     ↓
✂️ Split Data: 70% train, 30% test
     ↓
🤖 Build Model: GaussianNB classifier
     ↓
🎓 Train: Fit on training data
     ↓
🎯 Predict: Single & batch predictions
     ↓
📊 Evaluate: Accuracy, F1, Confusion Matrix
```

---

## 💡 Naive Bayes Quick Theory

**What is Naive Bayes?**

- Uses **Bayes' Theorem**: P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
- Assumes features are **independent** (naive assumption)
- Fast, simple, works well with small datasets

**Bayes Theorem Formula:**
$$P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}$$

Where:

- P(C|X) = Probability of class C given features X
- P(X|C) = Probability of features given class C
- P(C) = Prior probability of class C
- P(X) = Total probability of features

---

## 🔢 Key Variables & Meanings

| Variable          | Meaning                                 |
| ----------------- | --------------------------------------- |
| `x, y`            | Features and target labels              |
| `X, y`            | Feature matrix and target column        |
| `x_train, x_test` | Training and testing features           |
| `y_train, y_test` | Training and testing labels             |
| `model`           | Trained Naive Bayes classifier          |
| `predicted`       | Model predictions on test data          |
| `y_pred`          | All predictions on test set             |
| `accuracy`        | Correct predictions / Total predictions |
| `f1`              | F1 score (precision-recall balance)     |
| `cm`              | Confusion matrix (actual vs predicted)  |

---

## 🔄 Quick Code Examples

### **File 1: Multi-class Naive Bayes (Simple)**

```python
# 1. Generate Data
from sklearn.datasets import make_classification
x, y = make_classification(n_features=6, n_classes=3, n_samples=800,
                          n_informative=2, random_state=1)

# 2. Split Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

# 3. Build & Train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

# 4. Predict
predicted = model.predict([x_test[6]])
print(f"Actual: {y_test[6]}, Predicted: {predicted[0]}")

# 5. Evaluate
from sklearn.metrics import accuracy_score, f1_score
y_pred = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
```

### **File 2: Binary Classification (Real Data)**

```python
# 1. Load Data
import pandas as pd
df = pd.read_csv('Advertising.csv')
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

# 2. Prepare
df['SalesClass'] = ['High' if x > 15 else 'Low' for x in df['sales']]
X = df[['TV', 'radio', 'newspaper']]
y = df['SalesClass']

# 3. Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 4. Build & Train
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Predict & Evaluate
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
cm = confusion_matrix(y_test, y_pred, labels=['High', 'Low'])
print(cm)
```

---

## 📊 Comparison Table

| Aspect            | NaiveBayes.ipynb               | NaiveBayesEx.ipynb                |
| ----------------- | ------------------------------ | --------------------------------- |
| **Dataset**       | Synthetic (800 samples)        | Real (Advertising data)           |
| **Features**      | 6 numeric                      | 3 numeric (TV, radio, newspaper)  |
| **Classes**       | 3 (multiclass)                 | 2 (binary: High/Low)              |
| **Preprocessing** | None                           | Numeric conversion + Thresholding |
| **Data Type**     | Pure numeric                   | CSV file                          |
| **Visualization** | Scatter plot                   | Confusion matrix                  |
| **Metrics**       | Accuracy, F1, Confusion Matrix | Accuracy, F1, Confusion Matrix    |
| **Use Case**      | Learning basics                | Real-world application            |
| **Complexity**    | ⭐ Beginner                    | ⭐⭐ Intermediate                 |

---

## 🎯 Confusion Matrix Explained

```
                    Predicted
                 Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN

TP (True Positive):   Correctly predicted positive
FN (False Negative):  Missed positive cases
FP (False Positive):  Incorrectly predicted positive
TN (True Negative):   Correctly predicted negative
```

**Accuracy** = (TP + TN) / Total
**Precision** = TP / (TP + FP)
**Recall** = TP / (TP + FN)
**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

## 🚀 Complete Workflow Comparison

### **Multi-class (File 1)**

```
Synthetic Data (800×6)
        ↓
Train/Test Split (67/33)
        ↓
GaussianNB Model
        ↓
Predict Class (0, 1, or 2)
        ↓
Evaluate: Accuracy & F1
```

### **Binary (File 2)**

```
CSV Data
        ↓
Numeric Conversion
        ↓
Thresholding (>15 → High)
        ↓
Train/Test Split (70/30)
        ↓
GaussianNB Model
        ↓
Predict Class (High or Low)
        ↓
Evaluate: Accuracy, F1, Confusion Matrix
```

---

## 💡 Key Concepts to Remember

### **Gaussian Naive Bayes**

- Assumes features follow **normal distribution**
- Formula: P(x|C) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
- Best for: Continuous numeric data

### **How It Works (3 Steps)**

1. **Calculate Prior**: P(Class) = Count(Class) / Total
2. **Calculate Likelihood**: P(Feature|Class) for each feature
3. **Apply Bayes**: Multiply probabilities and pick highest

### **Advantages**

✅ Fast training & prediction
✅ Works with small datasets
✅ Handles multiclass problems
✅ No hyperparameter tuning needed
✅ Probabilistic predictions

### **Disadvantages**

❌ Assumes feature independence (often false)
❌ Ignores feature interactions
❌ Poor with high-dimensional data
❌ Not suitable for complex patterns

---

## 🔢 Example Predictions

### **File 1 (Multi-class)**

```
Input: x_test[6] with 6 features
Output: Class 0, 1, or 2
Example: Actual: 1, Predicted: 1 ✓
```

### **File 2 (Binary)**

```
Input: TV=100, Radio=50, Newspaper=30
Output: High or Low
Example: Actual: High, Predicted: High ✓
```

---

## 📈 Performance Metrics

| Metric    | Range | Interpretation                      |
| --------- | ----- | ----------------------------------- |
| Accuracy  | 0-1   | Overall correctness                 |
| Precision | 0-1   | Correctness of positive predictions |
| Recall    | 0-1   | Coverage of actual positives        |
| F1 Score  | 0-1   | Balance of precision & recall       |

**Typical Results:**

- NaiveBayes.ipynb: 85-95% accuracy
- NaiveBayesEx.ipynb: 80-90% accuracy

---

## 🔧 Installation & Setup

```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib
```

**Required Imports:**

```python
import pandas as pd              # Data manipulation
import numpy as np              # Numerical computing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt  # Visualization
```

---

## 🎓 Step-by-Step Learning Path

```
Start
  ↓
1️⃣ Understand Bayes' Theorem
  ↓
2️⃣ Run NaiveBayes.ipynb (synthetic data)
   ✅ See how model trains
   ✅ Learn accuracy calculation
  ↓
3️⃣ Run NaiveBayesEx.ipynb (real data)
   ✅ Handle CSV files
   ✅ Preprocess data
   ✅ Interpret confusion matrix
  ↓
4️⃣ Advanced Topics
   → Try different datasets
   → Experiment with train/test ratios
   → Compare with other classifiers
```

---

## ⚡ Quick Start Commands

**Run everything in order:**

```python
# File 1: NaiveBayes.ipynb
# Just run all cells top to bottom

# File 2: NaiveBayesEx.ipynb
# Make sure 'Advertising.csv' is in same directory
# Then run all cells top to bottom
```

---

## 🔄 Workflow Diagram

```
Both Files Follow Same Pattern:

┌─────────────────┐
│  Load/Generate  │
│     Data        │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Prepare Data   │
│  (optional)     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Train/Test Split│
└────────┬────────┘
         ↓
┌─────────────────┐
│  Build Model    │
│  GaussianNB()   │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Train: fit()    │
└────────┬────────┘
         ↓
┌─────────────────┐
│Predict: predict│
└────────┬────────┘
         ↓
┌─────────────────┐
│    Evaluate     │
│Accuracy, F1, CM │
└─────────────────┘
```

---

## 📋 Decision Tree: Which File to Use?

```
Do you have:
    ├─ Synthetic data, 3+ classes?
    │  └─ Use: NaiveBayes.ipynb ✓
    │
    └─ Real CSV data, binary classification?
       └─ Use: NaiveBayesEx.ipynb ✓
```

---

## ✅ Key Takeaways

🧠 **Naive Bayes Basics**

- Simple probabilistic classifier
- Assumes feature independence
- Fast and effective

📊 **When to Use**

- Text classification (spam detection)
- Sentiment analysis
- Medical diagnosis
- Email filtering
- Document categorization

⚙️ **Hyperparameters**

- GaussianNB has **no hyperparameters** to tune!
- Simplest scikit-learn classifier

🎯 **Best Practices**

1. Check for missing values
2. Ensure features are numeric (or encode them)
3. Split data before training
4. Use appropriate train/test ratio (80/20 or 70/30)
5. Evaluate with multiple metrics (not just accuracy)

---

## 📚 Formula Reference

**Bayes' Theorem:**
$$P(Class|Features) = \frac{P(Features|Class) \times P(Class)}{P(Features)}$$

**Gaussian Distribution:**
$$P(x|C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x-\mu_C)^2}{2\sigma_C^2}\right)$$

**Accuracy:**
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**F1 Score:**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 🎯 Quick Reference: Most Important Code Snippets

```python
# THE ESSENTIALS - Copy & Paste

# Train a Naive Bayes model
model = GaussianNB()
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Get accuracy
accuracy = accuracy_score(y_test, predictions)

# Get F1 score
f1 = f1_score(y_test, predictions, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
```

---

## 📞 Troubleshooting

| Problem             | Solution                                  |
| ------------------- | ----------------------------------------- |
| ModuleNotFoundError | `pip install scikit-learn`                |
| FileNotFoundError   | Ensure CSV in same directory              |
| Bad predictions     | Check data preprocessing                  |
| NaN values          | Use `pd.to_numeric(..., errors='coerce')` |
| Imbalanced accuracy | Check class distribution                  |

---

## 🏆 Performance Summary

| Aspect           | Score                    |
| ---------------- | ------------------------ |
| Speed            | ⚡⚡⚡⚡⚡ (Very Fast)   |
| Accuracy         | ⭐⭐⭐⭐ (Good)          |
| Simplicity       | ⭐⭐⭐⭐⭐ (Very Simple) |
| Real-world Use   | ⭐⭐⭐⭐ (Common)        |
| Interpretability | ⭐⭐⭐⭐⭐ (Easy)        |

---

**Created for EL 4152 - Machine Learning | Day 09 Practicals 🎓**

Last Updated: November 2025
