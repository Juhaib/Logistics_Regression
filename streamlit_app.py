# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import streamlit as st
import joblib

# Load the dataset
train_df = pd.read_csv('Titanic_train.csv')
test_df = pd.read_csv('Titanic_test.csv')

# Step 1: Data Exploration
print("Initial DataFrame Info:")
print(train_df.info())  # Check the info of the dataframe
print("Initial DataFrame Summary Statistics:")
print(train_df.describe())  # Get summary statistics

# Step 2: Data Preprocessing
# Drop unnecessary columns that contain string data
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)  # Handle missing Fare as well
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

# Check the processed DataFrame
print("Processed DataFrame:")
print(train_df.head())  # Print the first few rows of the processed DataFrame
print("Processed DataFrame Info:")
print(train_df.info())  # Check that all columns are numeric now

# Separate features and target variable
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

# Verify all features are numeric
print("Features Data Types:")
print(X.dtypes)

# Step 3: Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)

try:
    model.fit(X_train, y_train)
except ValueError as e:
    print("ValueError during model fitting:", e)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Display model evaluation metrics
print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Step 5: Interpretation
coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print("Logistic Regression Coefficients:")
print(coefficients)
significant_features = coefficients[coefficients['Coefficient'] != 0]
print('Significant features:')
print(significant_features)

# Step 6: Deployment with Streamlit
joblib.dump(model, 'titanic_model.pkl')
model = joblib.load('titanic_model.pkl')

# Streamlit app
st.title('Titanic Survival Predictor')

# User input
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, value=30)  # Default value for age
fare = st.number_input('Fare', min_value=0, value=10)  # Default value for fare
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Fare': [fare],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_C': [1 if embarked == 'C' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

# Align input data with the model's training features
input_data = input_data.reindex(columns=X.columns, fill_value=0)  # Ensure all columns are present

if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write('Survival Prediction: ', 'Survived' if prediction[0] == 1 else 'Did not survive')
