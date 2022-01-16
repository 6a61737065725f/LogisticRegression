# %%
import pandas as pd
import numpy as py
import time as time
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss, confusion_matrix, f1_score, accuracy_score
from sklearn.impute import SimpleImputer

start_time = time.time()
dataset = pd.read_csv("framingham.csv")
dataset.head()

preprocess = SimpleImputer(missing_values = py.NaN, strategy = 'mean')
dataset.totChol = preprocess.fit_transform(dataset['totChol'].values.reshape(-1, 1))[:,0]
dataset.BMI = preprocess.fit_transform(dataset['BMI'].values.reshape(-1, 1))[:,0]
dataset.heartRate = preprocess.fit_transform(dataset['heartRate'].values.reshape(-1, 1))[:,0]
dataset.glucose = preprocess.fit_transform(dataset['glucose'].values.reshape(-1, 1))[:,0]
dataset.age = preprocess.fit_transform(dataset['age'].values.reshape(-1, 1))[:,0]

dataset.isnull().sum()

input_features = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
x = dataset[input_features]
y = dataset.TenYearCHD

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logreg = LogisticRegression(max_iter = 10000)
logreg.fit(x_train, y_train)
y_pred_test = logreg.predict(x_test)
y_pred_train = logreg.predict(x_train)
y_pred_log_train = logreg.predict_proba(x_train)
y_pred_log_test = logreg.predict_proba(x_test)

# Test Data
print('*' * 100)
print('Test Data:\n', classification_report(y_test, y_pred_test))
print('Log Loss(Test Data):', log_loss(y_test, y_pred_log_test))
print('F1 Score:', 1 - f1_score(y_test, y_pred_test))
print('Accuracy Score:', accuracy_score(y_test, y_pred_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_test))
print('*' * 100)

# Train Data
print('Training Data:\n', classification_report(y_train, y_pred_train))
print('Log Loss(Training Data):', log_loss(y_train, y_pred_log_train))
print('F1 Score:', 1 - f1_score(y_train, y_pred_train))
print('Accuracy Score:', accuracy_score(y_train, y_pred_train))
print('Confusion Matrix:\n', confusion_matrix(y_train, y_pred_train))

# Some descriptions
print('*' * 100)
print('Coefficients/Weights:', logreg.coef_)
print('Intercept/Bias:', logreg.intercept_)
print('Iterations', logreg.n_iter_)


print('Run Time: ', round(time.time() - start_time, 10))
# %%
