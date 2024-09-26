# prompt: give an app.py file with all features which i can deploy to the streamlit cloud

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load your dataset
df = pd.read_csv('/content/fraud_oracle - fraud_oracle.csv')


# Create a LabelEncoder object
le = LabelEncoder()

# Iterate through the columns of the DataFrame
for column in df.columns:
  # Check if the column is of object type and has a limited number of unique values
  if df[column].dtype == 'object' and len(df[column].unique()) < 50:  # Adjust the threshold as needed
    try:
      # Fit and transform the column using LabelEncoder
      df[column] = le.fit_transform(df[column])
    except:
      print(f"Error encoding column: {column}")
      pass  # Handle the error as needed


y = df['FraudFound_P']
X = df.drop('FraudFound_P', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create individual classifiers
svm_clf = SVC(probability=True)
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()

# Create an ensemble using VotingClassifier
voting_clf = VotingClassifier(estimators=[('svm', svm_clf), ('knn', knn_clf), ('dt', dt_clf)], voting='soft')

# Train the ensemble model
voting_clf.fit(X_train, y_train)


# Streamlit app
st.title("Fraud Detection App")

# Sidebar for user input
st.sidebar.header("Enter Features:")

# Create input fields for each feature
input_features = {}
for column in X.columns:
  input_features[column] = st.sidebar.number_input(f"Enter {column}", value=0)


# Predict button
if st.sidebar.button("Predict"):
  # Create a DataFrame with user input
  user_input = pd.DataFrame([input_features])

  # Make prediction using the trained model
  prediction = voting_clf.predict(user_input)

  # Display the prediction
  st.write("## Prediction:")
  if prediction[0] == 1:
    st.error("Fraud Detected")
  else:
    st.success("No Fraud Detected")
