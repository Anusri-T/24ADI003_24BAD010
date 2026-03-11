print("Anusri T-24BAD010")
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
df=pd.read_csv(R"C:\Users\anusr\Downloads\StudentsPerformance.csv")
df.head()
le=LabelEncoder()

df['parental level of education']=le.fit_transform(df['parental level of education'])
df['test preparation course']=le.fit_transform(df['test preparation course'])
df['final_exam_score']=(df['math score'] + df['reading score'] + df['writing score'])/3
np.random.seed(42)

df['study_hours']=np.random.randint(1, 6, size=len(df))      
df['attendance']=np.random.randint(60, 100, size=len(df))    
df['sleep_hours']=np.random.randint(5, 9, size=len(df))      
df.fillna(df.mean(numeric_only=True), inplace=True)
X=df[['study_hours',
     'attendance',
     'parental level of education',
     'test preparation course',
     'sleep_hours']]

y=df['final_exam_score']
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
coefficients=pd.DataFrame({'Feature': X.columns,'Coefficient': model.coef_})

coefficients.sort_values(by='Coefficient', key=abs, ascending=False)

ridge=Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

ridge_pred=ridge.predict(X_test)
print("Ridge R²:", r2_score(y_test, ridge_pred))

lasso=Lasso(alpha=0.01)
lasso.fit(X_train, y_train)                            

lasso_pred=lasso.predict(X_test)
print("Lasso R²:", r2_score(y_test, lasso_pred))
residuals = y_test - y_pred

plt.figure()
plt.plot(y_test.values, label="Actual Score")
plt.plot(y_pred, label="Predicted Score")
plt.xlabel("Student Index")
plt.ylabel("Score")
plt.title("Actual vs Predicted Exam Scores")
plt.legend()
plt.show()


sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title("Feature Influence on Exam Score")
plt.show()

residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

print("Anusri T-24BAD010")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
df = pd.read_csv(R"C:\Users\anusr\Downloads\auto-mpg.csv")
df.replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'])
X = df[['horsepower']]
y = df['mpg']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
degrees = [2, 3, 4]

train_errors = []
test_errors = []
plt.figure(figsize=(10, 6))

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse_test)
    r2 = r2_score(y_test, y_test_pred)
    
    train_errors.append(mse_train)
    test_errors.append(mse_test)
    
    print(f"\nPolynomial Degree {d}")
    print("MSE:", mse_test)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    X_range_poly = poly.transform(X_range_scaled)
    y_range_pred = model.predict(X_range_poly)
    
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X_range, y_range_pred, label=f"Degree {d}")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression Curve Fitting")
plt.legend()
plt.show()
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Training vs Testing Error")
plt.legend()
plt.show()
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)

y_ridge_pred = ridge.predict(X_test_poly)

print("\nRidge Regression (Degree 4)")
print("MSE:", mean_squared_error(y_test, y_ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_ridge_pred)))
print("R2 Score:", r2_score(y_test, y_ridge_pred))
