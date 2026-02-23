SCENARIO 1: K-Nearest Neighbors (KNN) – Breast Cancer Classification

Dataset: Breast Cancer Dataset

Link:
https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Objective:
To predict whether a tumor is Benign or Malignant using the K-Nearest Neighbors (KNN) classification algorithm based on medical measurements.

Target Variable

Diagnosis (Benign / Malignant)

Input Features

Radius

Texture

Perimeter

Area

Smoothness

Tasks Performed:

Imported required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).

Loaded the Breast Cancer dataset into a Pandas DataFrame.

Inspected dataset using head(), info(), and describe().

Checked for missing values and handled inconsistencies.

Encoded target labels (Benign = 0, Malignant = 1).

Applied feature scaling using StandardScaler (important for KNN).

Split dataset into training and testing sets (80% – 20%).

Trained a KNN classifier.

Experimented with different values of K.

Predicted diagnosis labels for test data.

Evaluated performance using:

Accuracy

Precision

Recall

F1 Score

Identified misclassified cases.

Analyzed model sensitivity to different K values.

Visualizations:

Confusion Matrix for classification results.

Accuracy vs K plot to determine optimal K value.

Decision Boundary plot (using two selected features).

Outcome:

The KNN model successfully classified tumors as benign or malignant with high accuracy. Feature scaling significantly improved performance. The Accuracy vs K plot helped determine the optimal K value, and analysis showed that very small or very large K values may reduce model performance due to overfitting or underfitting.

SCENARIO 2: Decision Tree Classifier – Loan Prediction

Dataset: Loan Prediction Dataset

Link:
https://www.kaggle.com/datasets/ninzaami/loan-predication

Objective:
To predict whether a loan application should be Approved or Rejected using a Decision Tree Classifier.

Target Variable

Loan Status (Approved / Rejected)

Input Features

Applicant Income

Loan Amount

Credit History

Education

Property Area

Tasks Performed:

Imported required Python libraries.

Loaded the Loan Prediction dataset into Pandas.

Inspected dataset structure and checked for missing values.

Performed preprocessing:

Handled missing values

Encoded categorical variables using Label Encoding / One-Hot Encoding

Split dataset into training and testing sets.

Trained a Decision Tree classifier.

Experimented with tree depth and pruning techniques.

Predicted loan approval status.

Evaluated performance using:

Accuracy

Precision

Recall

F1 Score

Analyzed feature importance.

Detected overfitting behavior.

Compared shallow vs deep trees.

Visualizations:

Confusion Matrix to evaluate predictions.

Tree Structure Plot for model interpretation.

Feature Importance Plot to identify key decision factors.

Outcome:

The Decision Tree model effectively predicted loan approval status. Credit History and Applicant Income were found to be major influencing factors. Shallow trees generalized better, while deeper trees showed signs of overfitting. Feature importance analysis provided interpretability to the model’s decision-making process
