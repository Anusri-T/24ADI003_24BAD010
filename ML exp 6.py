print("24BAD010 - Anusri T")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv(r"C:\Users\anusr\Downloads\diabetes_bagging.csv")
print(df.head())
X = df[['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", dt_accuracy)
bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=50,random_state=42)
bag_model.fit(X_train, y_train)
y_pred_bag = bag_model.predict(X_test)
bag_accuracy = accuracy_score(y_test, y_pred_bag)
print("Bagging Accuracy:", bag_accuracy)
models = ['Decision Tree', 'Bagging']
accuracies = [dt_accuracy, bag_accuracy]
plt.figure()
plt.bar(models, accuracies)
plt.title("Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()
cm = confusion_matrix(y_test, y_pred_bag)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Bagging")
plt.show()


print("24BAD010 - Anusri T")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\anusr\Downloads\churn_boosting.csv")

le = LabelEncoder()

df['ContractType'] = le.fit_transform(df['ContractType'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['Churn'] = le.fit_transform(df['Churn'])

X = df[['Tenure', 'MonthlyCharges', 'ContractType', 'InternetService']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

ada_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_prob_ada = ada_model.predict_proba(X_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
auc_ada = auc(fpr_ada, tpr_ada)

y_prob_gb = gb_model.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
auc_gb = auc(fpr_gb, tpr_gb)

print("AdaBoost AUC:", auc_ada)
print("Gradient Boosting AUC:", auc_gb)

plt.figure()
plt.plot(fpr_ada, tpr_ada, label="AdaBoost")
plt.plot(fpr_gb, tpr_gb, label="Gradient Boosting")
plt.plot([0,1], [0,1], linestyle='--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

features = X.columns

plt.figure()
plt.bar(features, ada_model.feature_importances_)
plt.title("Feature Importance - AdaBoost")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

plt.figure()
plt.bar(features, gb_model.feature_importances_)
plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

print("24BAD010 - Anusri T")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\anusr\Downloads\income_random_forest.csv")

print(df.head())

le = LabelEncoder()
df['Income'] = le.fit_transform(df['Income'])

X = df[['Age', 'EducationYears', 'HoursPerWeek', 'Experience']]
y = df['Income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Initial Accuracy:", accuracy)

tree_range = [10, 20, 50, 100, 150]
accuracies = []

for n in tree_range:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure()
plt.plot(tree_range, accuracies, marker='o')
plt.title("Accuracy vs Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.show()

importances = rf_model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

print("24BAD010 - Anusri T")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\anusr\Downloads\heart_stacking.csv")

X = df[['Age', 'Cholesterol', 'MaxHeartRate', 'RestingBP']]
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000)
svm = SVC(probability=True)
dt = DecisionTreeClassifier(random_state=42)

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_dt = accuracy_score(y_test, y_pred_dt)

estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('svm', SVC(probability=True)),
    ('dt', DecisionTreeClassifier())
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)

models = ['Logistic Regression', 'SVM', 'Decision Tree', 'Stacking']
accuracies = [acc_lr, acc_svm, acc_dt, acc_stack]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


print("24BAD010 - Anusri T")
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\anusr\Downloads\fraud_smote.csv")

X = df[['Amount', 'Time', 'Feature1', 'Feature2']]
y = df['Fraud']

class_counts = y.value_counts()

plt.figure()
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)

y_prob_before = model_before.predict_proba(X_test)[:, 1]
precision_before, recall_before, _ = precision_recall_curve(y_test, y_prob_before)
pr_auc_before = auc(recall_before, precision_before)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

class_counts_after = pd.Series(y_train_sm).value_counts()

plt.figure()
plt.bar(class_counts_after.index.astype(str), class_counts_after.values)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

model_after = LogisticRegression(max_iter=1000)
model_after.fit(X_train_sm, y_train_sm)

y_prob_after = model_after.predict_proba(X_test)[:, 1]
precision_after, recall_after, _ = precision_recall_curve(y_test, y_prob_after)
pr_auc_after = auc(recall_after, precision_after)

plt.figure()
plt.plot(recall_before, precision_before, label=f"Before SMOTE (AUC={pr_auc_before:.2f})")
plt.plot(recall_after, precision_after, label=f"After SMOTE (AUC={pr_auc_after:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
