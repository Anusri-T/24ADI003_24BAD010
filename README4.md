SCENARIO 1: Multinomial Naïve Bayes – SMS Spam Classification

Dataset: SMS Spam Collection Dataset

Link:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Objective:
To classify SMS messages as Spam or Ham (Not Spam) using the Multinomial Naïve Bayes algorithm and analyze text patterns influencing spam detection.

Tasks Performed:

Imported required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).

Loaded the SMS Spam dataset into a Pandas DataFrame.

Inspected dataset structure using head(), info(), and describe().

Performed text preprocessing:

Converted text to lowercase

Removed punctuation and special characters

Removed stopwords (optional)

Converted text into numerical features using:

Count Vectorization

TF-IDF Vectorization

Encoded target labels (Ham = 0, Spam = 1).

Split dataset into training and testing sets.

Trained a Multinomial Naïve Bayes classifier.

Predicted message classes for test data.

Evaluated model using Accuracy, Precision, Recall, and F1 Score.

Analyzed misclassified messages.

Applied Laplace smoothing (alpha parameter) and compared results.

Visualized:

Confusion Matrix for classification performance.

Feature importance (Top words influencing spam classification).

Word frequency comparison between Spam and Ham messages.

Outcome:

The analysis showed that Multinomial Naïve Bayes effectively classifies SMS messages with high accuracy. Spam messages commonly contain promotional keywords such as "free", "win", and "call". Laplace smoothing improved model generalization by handling zero probability issues. Misclassification analysis helped identify overlapping vocabulary between Spam and Ham messages.

SCENARIO 2: Gaussian Naïve Bayes – Iris Flower Classification

Dataset: Iris Dataset

Source: Scikit-learn Built-in Dataset

Objective:
To classify flower species based on physical measurements using the Gaussian Naïve Bayes algorithm and evaluate model performance.

Target Variable

Flower Species

Input Features

Sepal Length

Sepal Width

Petal Length

Petal Width

Tasks Performed:

Imported required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn).

Loaded the Iris dataset from sklearn.

Converted dataset into a Pandas DataFrame for easier analysis.

Inspected dataset structure using head(), info(), and describe().

Checked for missing values and verified data types.

Applied feature scaling using StandardScaler.

Split dataset into training and testing sets (80% – 20%).

Trained a Gaussian Naïve Bayes classifier.

Predicted species labels on test data.

Evaluated performance using:

Accuracy

Precision

Recall

F1 Score

Compared predicted labels with actual labels.

Analyzed class probabilities using predict_proba().

Compared Gaussian Naïve Bayes with Logistic Regression (optional analysis).

Visualizations:

Decision Boundary Plot (using two selected features).

Confusion Matrix to evaluate classification results.

Probability distribution plots for feature analysis.

Outcome:

The Gaussian Naïve Bayes model successfully classified iris flower species with high accuracy. Petal length and petal width were found to be the most influential features in classification. The confusion matrix showed minimal misclassification among species. Probability analysis helped understand model confidence in predictions. Comparison with Logistic Regression demonstrated similar performance, validating the effectiveness of Gaussian Naïve Bayes for normally distributed data.
