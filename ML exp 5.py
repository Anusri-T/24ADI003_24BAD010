print("24BAD010-Anusri T")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = pd.read_csv(R"C:\Users\anusr\Downloads\breast-cancer.csv")

features = ['radius_mean', 'texture_mean', 
            'perimeter_mean', 'area_mean', 
            'smoothness_mean']

X = data[features]
y = data['diagnosis']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=encoder.classes_))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – KNN")
plt.show()

misclassified_indices = np.where(y_test != y_pred)[0]
print("Number of Misclassified Cases:", len(misclassified_indices))

k_values = range(1, 21)
accuracy_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, pred))

plt.figure()
plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K")
plt.show()

best_k = k_values[np.argmax(accuracy_list)]
print("Best K:", best_k)

X_2d = X_scaled[:, :2]
y_2d = y_encoded

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d,
    test_size=0.25,
    random_state=42,
    stratify=y_2d
)

knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_2d)
plt.xlabel("Radius (scaled)")
plt.ylabel("Texture (scaled)")
plt.title("Decision Boundary – KNN")
plt.show()


print("24BAD010-Anusri T")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\anusr\Downloads\train_u6lujuX_CVtuZ9i (1).csv")

data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['ApplicantIncome'] = data['ApplicantIncome'].fillna(data['ApplicantIncome'].median())
data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
data['Property_Area'] = data['Property_Area'].fillna(data['Property_Area'].mode()[0])
data['Loan_Status'] = data['Loan_Status'].fillna(data['Loan_Status'].mode()[0])

features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Education', 'Property_Area']
X = data[features]
y = data['Loan_Status']

X = pd.get_dummies(X, drop_first=True)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Decision Tree")
plt.show()

plt.figure(figsize=(15,8))
plot_tree(dt, feature_names=X.columns, class_names=encoder.classes_, filled=True)
plt.show()

importances = pd.Series(dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure()
importances.plot(kind='bar')
plt.title("Feature Importance")
plt.show()

depth_values = range(1, 21)
train_scores = []
test_scores = []

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(depth_values, train_scores)
plt.plot(depth_values, test_scores)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Shallow vs Deep Tree Performance")
plt.legend(["Train Accuracy", "Test Accuracy"])
plt.show()
