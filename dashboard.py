import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import numpy as np

# Load data
df = pd.read_csv("exercice_data.csv", sep=None, encoding='latin1')

# Preprocessing
replace_dict = {"yes": 1, "no": 0}
df.replace(replace_dict, inplace=True)

# Define columns to apply one-hot encoding
columns_to_encode = ['sex', 'address', 'famsize', 'Pstatus', 'reason', 'guardian', 'Mjob', 'Fjob']

# Define transformer for one-hot encoding
encoder = OneHotEncoder(drop='first')

# Define preprocessor for the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', encoder, columns_to_encode)
    ],
    remainder='passthrough'
)

# Define pipeline with preprocessor and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        random_state=42,
        n_estimators=100,  
        max_depth=2,  
        learning_rate=0.1,  
        min_child_weight=1,  
        subsample=0.8,  
        colsample_bytree=0.8,  
        reg_alpha=0,  
        reg_lambda=1,  
        gamma=0,  
        ))
])

# Split data into features and target variable
X = df.drop(columns=['FinalGrade', 'StudentID', 'FirstName', 'FamilyName'])
y = df['FinalGrade'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model on the training set
pipeline.fit(X_train, y_train)

# Retrieve feature importance
feature_importance = pipeline.named_steps['model'].feature_importances_

# Sort feature importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]

# Select top N features
top_features = sorted_idx[:15]  

# Retrieve feature names after one-hot encoding
one_hot_encoded_feature_names = preprocessor.transformers_[0][1].get_feature_names_out(input_features=columns_to_encode)
remaining_feature_names = [col for col in X.columns if col not in columns_to_encode]
feature_names = list(one_hot_encoded_feature_names) + remaining_feature_names

# Make predictions on both train and test sets
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Calculate evaluation metrics for train set
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

# Calculate evaluation metrics for test set
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Create DataFrame with metrics
metrics_df = pd.DataFrame({
    'Set': ['Train', 'Test'],
    'RMSE': [train_rmse, test_rmse],
    'MAE': [train_mae, test_mae],
    'R2': [train_r2, test_r2],
    'MSE': [train_mse, test_mse]
})

# Compute the composite score
numerical_data = df.select_dtypes(include=['number'])
max_values = numerical_data.max()
min_values = numerical_data.min()
normalized_data = (numerical_data - min_values) / (max_values - min_values)
weights = dict(zip([feature_names[i] for i in top_features], feature_importance[top_features]))
total_importance = sum(weights.values())
for feature in weights:
    weights[feature] /= total_importance
composite_score = (normalized_data * pd.Series(weights)).sum(axis=1)
df['composite_score'] = composite_score

# Streamlit app
st.title("Performance Dashboard")

# Display feature importance
st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(top_features)), feature_importance[top_features], align="center")
ax.set_xticks(range(len(top_features)))
ax.set_xticklabels([feature_names[i] for i in top_features], rotation=45, ha='right')
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
st.pyplot(fig)

# Display scatter plot
st.subheader("Composite Score vs Final Grade")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['FinalGrade'], df['composite_score']*10, alpha=0.5)
ax.set_title('Composite Score vs Final Grade')
ax.set_ylabel('Composite Score')
ax.set_xlabel('Final Grade')
ax.grid(True)
st.pyplot(fig)

# Display evaluation metrics table
st.subheader("Evaluation Metrics")
st.table(metrics_df)

