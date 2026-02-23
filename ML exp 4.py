print("24BAD010-Anusri T")
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv(r"C:\Users\anusr\Downloads\spam.csv", encoding="latin-1")


data = data[['v1', 'v2']]
data.columns = ['label', 'message']

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

data['clean_message'] = data['message'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['clean_message'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print("-----------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test_df = data.loc[idx_test].copy()

test_df['Actual'] = label_encoder.inverse_transform(y_test)
test_df['Predicted'] = label_encoder.inverse_transform(y_pred)

misclassified = test_df[test_df['Actual'] != test_df['Predicted']]

print("\nMISCLASSIFIED EXAMPLES")
print("----------------------")
print(misclassified[['message', 'Actual', 'Predicted']].head())

print("\nLAPLACE SMOOTHING IMPACT")
print("-----------------------")
alphas = [0.01, 0.1, 1, 5]
for a in alphas:
    nb = MultinomialNB(alpha=a)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)
    print(f"Alpha={a} → Accuracy={accuracy_score(y_test, preds):.4f}")


cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = np.array(vectorizer.get_feature_names_out())
spam_log_prob = model.feature_log_prob_[1]

top_spam_indices = np.argsort(spam_log_prob)[-15:]
top_spam_words = feature_names[top_spam_indices]
top_spam_values = spam_log_prob[top_spam_indices]

plt.figure()
plt.barh(top_spam_words, top_spam_values)
plt.xlabel("Log Probability")
plt.title("Top Words Influencing Spam Classification")
plt.show()

spam_messages = data[data['label'] == 'spam']['clean_message']
ham_messages = data[data['label'] == 'ham']['clean_message']

spam_vec = CountVectorizer(max_features=10)
ham_vec = CountVectorizer(max_features=10)

spam_counts = spam_vec.fit_transform(spam_messages).toarray().sum(axis=0)
ham_counts = ham_vec.fit_transform(ham_messages).toarray().sum(axis=0)

plt.figure()
plt.bar(spam_vec.get_feature_names_out(), spam_counts)
plt.title("Top Word Frequencies in Spam Messages")
plt.show()

plt.figure()
plt.bar(ham_vec.get_feature_names_out(), ham_counts)
plt.title("Top Word Frequencies in Ham Messages")
plt.show()


# SCENARIO 2 – GAUSSIAN NAÏVE BAYES
# Iris Flower Classification
print("24BAD010-Anusri T")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

iris = load_iris()

X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMODEL PERFORMANCE (Gaussian NB)")
print("--------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Gaussian Naïve Bayes")
plt.show()

comparison = pd.DataFrame({
    'Actual': class_names[y_test],
    'Predicted': class_names[y_pred]
})

print("\nPrediction Comparison (First 10):")
print(comparison.head(10))

y_prob = gnb.predict_proba(X_test)

prob_df = pd.DataFrame(y_prob, columns=class_names)
print("\nClass Probabilities (First 5 Samples):")
print(prob_df.head())

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLOGISTIC REGRESSION PERFORMANCE")
print("--------------------------------")
print(classification_report(y_test, lr_pred, target_names=class_names))

X_2d = X_scaled[:, 2:4]   
y_2d = y

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.25, random_state=42, stratify=y_2d
)

gnb_2d = GaussianNB()
gnb_2d.fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = gnb_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_2d, palette='Set1')
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Decision Boundary – Gaussian Naïve Bayes")
plt.show()

plt.figure()
for i, cls in enumerate(class_names):
    sns.kdeplot(y_prob[:, i], label=cls)

plt.xlabel("Predicted Probability")
plt.title("Class Probability Distribution – Gaussian NB")
plt.legend()
plt.show()
