

# 🔍 Rock vs Mine Classification — Logistic Regression Model

This repository contains a machine learning project that implements a **binary classification model** to distinguish between sonar signals reflected from **underwater mines (M)** and **rocks (R)** using frequency-based signal data. The model is developed using Python and scikit-learn, and is trained using **Logistic Regression**, a widely used linear model suitable for binary classification tasks.

---

## 📂 Project Structure

* `Rock vs mine prediction.ipynb` — Jupyter Notebook containing the full implementation including data preprocessing, training, evaluation, and prediction.

---

## 📌 Problem Statement

The objective is to build a **predictive model** that can accurately classify whether a sonar return is from a **metal cylinder (mine)** or a **rock**, based on 60 frequency amplitude values per observation. The dataset used is the **Sonar Dataset** from the UCI Machine Learning Repository.

---

## 🧰 Technologies and Libraries Used

* Python 3.x
* NumPy
* Pandas
* Scikit-learn (sklearn)

  * `LogisticRegression`
  * `train_test_split`
  * `accuracy_score`

---

## 📊 Dataset Overview

* **Instances:** 208
* **Features:** 60 numerical attributes representing sonar signal amplitudes
* **Target column:** `Label (M or R)`
* **Source:** UCI Machine Learning Repository

---

## 🧪 Data Preprocessing

* The dataset was loaded using `pandas.read_csv` with `header=None` as the dataset lacks column names.
* Features (`X`) were extracted by dropping column index `60`.
* Target labels (`y`) were extracted from column index `60`, which contains binary categorical values: `'M'` for mine and `'R'` for rock.
* The dataset was split using `train_test_split`:

  * **Train size:** 90%
  * **Test size:** 10%
  * **Stratified sampling:** Enabled via `stratify=y` to preserve class distribution
  * **Random seed:** `random_state=1` to ensure reproducibility

---

## 🧠 Model Training

* Algorithm Used: **Logistic Regression**
* Reason: Logistic Regression is suitable for binary classification and interpretable due to its linear nature.
* Training was done on the `x_train`, `y_train` sets using `model.fit()`.

---

## 📈 Model Evaluation

* Predictions were made on both training and testing datasets.
* Accuracy scores were calculated using `accuracy_score()`:

  * **Training Accuracy:** \~83.42%
  * **Testing Accuracy:** \~76.19%

> Note: A slight drop in accuracy between training and test sets is expected and indicates generalization performance.

---

## 🔮 Prediction System

The notebook also includes a section that allows real-time prediction using user-provided sonar frequency values (60 features). These values are reshaped into the required input format for the trained model, and the prediction (`M` or `R`) is returned accordingly.

```python
# Input (example)
input_data = (0.0762, 0.0666, 0.0481, ..., 0.0094)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
```

---

## 📌 Key Takeaways

* Demonstrated end-to-end machine learning pipeline from data ingestion to prediction.
* Used **Logistic Regression** as a baseline classification model.
* Achieved decent accuracy despite small dataset size.
* Maintained clean, modular code with commentary for clarity.

---

## 🚫 Disclaimer

🚨 *Note*: The project contains placeholder print statements with informal and non-professional language meant only for debugging or entertainment purposes during local development. These should be **removed or replaced** before deployment or production usage to maintain professionalism and appropriateness.

---

## ✅ Future Improvements

* Implement advanced models like SVM, Random Forest, or Neural Networks for performance benchmarking.
* Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Apply feature scaling (e.g., `StandardScaler`) to further optimize logistic regression results.
* Incorporate confusion matrix, precision, recall, and ROC-AUC score for better evaluation metrics.

---


