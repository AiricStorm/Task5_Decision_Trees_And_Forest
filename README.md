# 🌲 Task 5: Decision Trees and Random Forests

This project demonstrates the use of **tree-based classification models** using the **Heart Disease Dataset** from Kaggle. We compare a single **Decision Tree** to an ensemble-based **Random Forest**, evaluate their performance, and visualize key aspects of the models.

---

## 🎯 Objective

- Understand how Decision Trees and Random Forests work.
- Control overfitting by pruning or limiting tree depth.
- Interpret model predictions using feature importance.
- Use cross-validation for reliable performance estimation.

---

## 📊 Dataset

- **Source**: [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Target Variable**: `target` (0 = no heart disease, 1 = heart disease)
- **Features**: 13 numerical and categorical medical features (e.g., age, cholesterol, chest pain type)

---

## 🔧 Workflow

### 1. Data Preprocessing

- Removed nulls (none present in this dataset).
- Split dataset into `X` (features) and `y` (target).
- Used `train_test_split()` to divide into 80% training and 20% test sets.

### 2. Decision Tree Classifier

- Trained a `DecisionTreeClassifier`.
- Controlled overfitting with `max_depth`.
- Visualized the tree using `plot_tree`.

### 3. Random Forest Classifier

- Trained a `RandomForestClassifier` with 100 trees.
- Compared its performance with the Decision Tree.
- Analyzed **feature importance**.

### 4. Evaluation

- Used accuracy, confusion matrix, and classification report.
- Applied 5-fold **cross-validation** for reliability.

---

## ✅ Results Summary

| Model          | Accuracy | Cross-Val Accuracy | Comments                     |
|----------------|----------|--------------------|------------------------------|
| Decision Tree  | ~85–88%  | ~83–85%             | Prone to overfitting         |
| Random Forest  | ~90–94%  | ~88–91%             | More stable and accurate     |

### 🔍 Feature Importance (Top 5)

1. `cp` – chest pain type  
2. `thalach` – maximum heart rate  
3. `exang` – exercise-induced angina  
4. `oldpeak` – ST depression  
5. `ca` – number of major vessels

---

## 📈 Visuals

- **Decision Tree**: Clear branching logic.
- **Confusion Matrix**: Compared predictions across both models.
- **Feature Importance Bar Plot**: Explained most influential predictors.

---

## 🧠 Takeaways

- **Decision Trees** are interpretable but may overfit.
- **Random Forests** improve generalization and are more robust.
- **Cross-validation** is essential for reliable evaluation.
- Feature importance helps understand model decisions.

---

## 🛠 Tools Used

- Python
- Pandas, Matplotlib, Seaborn
- Scikit-learn (DecisionTreeClassifier, RandomForestClassifier, cross_val_score)

---

## 📁 Folder Structure

├── data/
  └── heart.csv
├──visualizations/
  └──Decision_tree.png
  └──feature_Importance.png
  └──Random_forest.png
└── notebook/
  └── decision.py
├── README.md

## ✅ Submission

- Repository: [https://github.com/AiricStorm/Task4_Classification_Logistic_Regression]
- Submitted via: [Google Form Link](https://forms.gle/dqTJkbRFU7jYnkhD9)
